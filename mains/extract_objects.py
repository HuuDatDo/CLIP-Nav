import sys
import math
import spacy
import csv
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from learning.modules.key_tensor_store import KeyTensorStore

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from data_io.paths import get_logging_dir
from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params
from learning.dual_dataloader import DualDataloader

from utils.logging_summary_writer import LoggingSummaryWriter
from parameters.parameter_server import get_current_parameters
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.instructions import get_restricted_env_id_lists

import parameters.parameter_server as P


PROFILE = False


class TrainerBidomain:
    def __init__(
            self,
            model_real,
            model_sim,
            model_critic,
            state=None,
            epoch=0
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)
        self.nlp = spacy.load("en_core_web_sm")
        self.gpt3 = None

        self.params = get_current_parameters()["Training"]
        self.run_name = get_current_parameters()["Setup"]["run_name"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']
        self.critic_steps = self.params['critic_steps']
        self.critic_warmup_steps = self.params['critic_warmup_steps']
        self.critic_warmup_iterations = self.params['critic_warmup_iterations']
        self.model_steps = self.params['model_steps']
        self.critic_batch_size = self.params["critic_batch_size"]
        self.model_batch_size = self.params["model_batch_size"]
        self.disable_wloss = self.params["disable_wloss"]
        self.sim_steps_per_real_step = self.params.get("sim_steps_per_real_step", 1)
        self.real_grad_noise = self.params.get("real_grad_noise", False)

        self.critic_steps_cycle = self.params.get("critic_steps_cycle", False)
        self.critic_steps_amplitude = self.params.get("critic_steps_amplitude", 0)
        self.critic_steps_period = self.params.get("critic_steps_period", 1)

        self.sim_datasets = get_current_parameters()["Data"]["sim_datasets"]
        self.real_datasets = get_current_parameters()["Data"]["real_datasets"]

        n_params_real = get_n_params(model_real)
        n_params_real_tr = get_n_trainable_params(model_real)
        n_params_sim = get_n_params(model_sim)
        n_params_sim_tr = get_n_trainable_params(model_sim)
        n_params_c = get_n_params(model_critic)
        n_params_c_tr = get_n_params(model_critic)

        print("Training Model:")
        print("Real # model parameters: " + str(n_params_real))
        print("Real # trainable parameters: " + str(n_params_real_tr))
        print("Sim  # model parameters: " + str(n_params_sim))
        print("Sim  # trainable parameters: " + str(n_params_sim_tr))
        print("Critic  # model parameters: " + str(n_params_c))
        print("Critic  # trainable parameters: " + str(n_params_c_tr))

        # Share those modules that are to be shared between real and sim models
        if not self.params.get("disable_domain_weight_sharing"):
            print("Sharing weights between sim and real modules")
            model_real.steal_cross_domain_modules(model_sim)
        else:
            print("NOT Sharing weights between sim and real modules")

        self.model_real = model_real
        self.model_sim = model_sim
        self.model_critic = model_critic

        if self.optimizer == "adam":
            Optim = optim.Adam
        elif self.optimizer == "sgd":
            Optim = optim.SGD
        else:
            raise ValueError(f"Unsuppored optimizer {self.optimizer}")

        self.optim_models = Optim(self.model_real.both_domain_parameters(self.model_sim), self.lr, weight_decay=self.weight_decay)
        self.optim_critic = Optim(self.get_model_parameters(self.model_critic), self.lr, weight_decay=self.weight_decay)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)

    def get_model_parameters(self, model):
        params_out = []
        skipped_params = 0
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
        print(str(skipped_params) + " parameters frozen")
        return params_out

    def get_state(self):
        state = {}
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def train_epoch(self, env_list=None, data_list_real=None, data_list_sim=None, eval=False, restricted_domain=False):

        if eval:
            self.model_real.eval()
            self.model_sim.eval()
            self.model_critic.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model_real.train()
            self.model_sim.train()
            self.model_critic.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1

        # Allow testing with both domains being simulation domain
        if self.params["sim_domain_only"]:
            dataset_real = self.model_sim.get_dataset(data=data_list_sim, envs=env_list, dataset_names=self.sim_datasets, dataset_prefix="supervised", eval=eval)
            self.model_real = self.model_sim
        else:
            dataset_real = self.model_real.get_dataset(data=data_list_real, envs=env_list, dataset_names=self.real_datasets, dataset_prefix="supervised", eval=eval)

        dataset_sim = self.model_sim.get_dataset(data=data_list_sim, envs=env_list, dataset_names=self.sim_datasets, dataset_prefix="supervised", eval=eval)

        #Change shuffle to False so we can synchronize the original dataset and csv
        dual_model_loader = DualDataloader(
            dataset_a=dataset_real,
            dataset_b=dataset_sim,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="max"
        )

        dual_critic_loader = DualDataloader(
            dataset_a=dataset_real,
            dataset_b=dataset_sim,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="infinite"
        )
        dual_critic_iterator = iter(dual_critic_loader)

        #wloss_before_updates_writer = LoggingSummaryWriter(log_dir=f"runs/{self.run_name}/discriminator_before_updates")
        #wloss_after_updates_writer = LoggingSummaryWriter(log_dir=f"runs/{self.run_name}/discriminator_after_updates")

        samples_real = len(dataset_real)
        samples_sim = len(dataset_sim)
        if samples_real == 0 or samples_sim == 0:
            print (f"DATASET HAS NO DATA: REAL: {samples_real > 0}, SIM: {samples_sim > 0}")
            return -1.0

        num_batches = len(dual_model_loader)

        epoch_loss = 0
        count = 0
        critic_elapsed_iterations = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        # Alternate training critic and model
        for real_batch, sim_batch in dual_model_loader:
            if restricted_domain == "real":
                sim_batch = real_batch
            elif restricted_domain == "simulator":
                real_batch = sim_batch
            if real_batch is None or sim_batch is None:
                continue

            # We run more updates on the sim data than on the real data to speed up training and
            # avoid overfitting on the scarce real data
            if self.sim_steps_per_real_step == 1 or self.sim_steps_per_real_step == 0 or count % self.sim_steps_per_real_step == 0:
                train_sim_only = False
            else:
                train_sim_only = True

            if sim_batch is None or (not train_sim_only and real_batch is None):
                continue

            prof.tick("load_model_data")
            # Train the model for model_steps in a row, then train the critic, and repeat
            critic_batch_num = 0

            if count % self.model_steps == 0 and not eval and not self.disable_wloss:

                    # Each batch is actually a single rollout (we batch the rollout data across the sequence)
                    # To collect a batch of rollouts, we need to keep iterating
                    real_store = KeyTensorStore()
                    sim_store = KeyTensorStore()
                    for b in range(self.critic_batch_size):
                        # Get the next non-None batch
                        real_c_batch, sim_c_batch = None, None
                        while real_c_batch is None or sim_c_batch is None:
                            real_c_batch, sim_c_batch = next(dual_critic_iterator)
                        prof.tick("critic_load_data")
                        # When training the critic, we don't backprop into the model, so we don't need gradients here
                        with torch.no_grad():
                            #Get images and instructions from batch of data
                            images = sim_c_batch["images"][0]
                            seq_len = len(images)
                            instructions = sim_c_batch["instr"][0][:seq_len]
                            
                            #Convert from tokenized list to text-instructions
                            textual_instructions = ""
                            for i in range(len(instructions[0])-1):
                                textual_instructions += self.token2word[int(instructions[0][i])]
                                textual_instructions += " "
                            
                            #Forward to GPT3 
                            object_list = []
                            for chunks in self.nlp(textual_instructions).noun_chunks:
                                prompt = "Is" + chunks + "an object?"
                                answer = self.gpt3(prompt)
                                if answer == "Yes":
                                    object_list.append(chunks)
                                    
                            #Write to csv
                            with open('.csv','a') as f_object:
                                writer_object = csv.writer(f_object)
                                writer_object.writerow([textual_instructions, object_list])
                                f_object.close()
                                
                                
                                
class TrainerBidomainBidata:
    def __init__(
            self,
            model_real,
            model_sim,
            model_critic,
            model_oracle_critic=None,
            state=None,
            epoch=0
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)
        self.nlp = spacy.load("en_core_web_lg")

        self.params = get_current_parameters()["Training"]
        self.run_name = get_current_parameters()["Setup"]["run_name"]
        self.batch_size = self.params['batch_size']
        self.iterations_per_epoch = self.params.get("iterations_per_epoch", None)
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.critic_loaders = self.params['critic_loaders']
        self.model_common_loaders = self.params['model_common_loaders']
        self.model_sim_loaders = self.params['model_sim_loaders']
        self.lr = self.params['lr']
        self.critic_steps = self.params['critic_steps']
        self.model_steps = self.params['model_steps']
        self.critic_batch_size = self.params["critic_batch_size"]
        self.model_batch_size = self.params["model_batch_size"]
        self.disable_wloss = self.params["disable_wloss"]
        self.sim_steps_per_real_step = self.params.get("sim_steps_per_real_step", 1)

        self.real_dataset_names = self.params.get("real_dataset_names")
        self.sim_dataset_names = self.params.get("sim_dataset_names")

        self.bidata = self.params.get("bidata", False)
        self.sim_steps_per_common_step = self.params.get("sim_steps_per_common_step", 1)

        n_params_real = get_n_params(model_real)
        n_params_real_tr = get_n_trainable_params(model_real)
        n_params_sim = get_n_params(model_sim)
        n_params_sim_tr = get_n_trainable_params(model_sim)
        n_params_c = get_n_params(model_critic)
        n_params_c_tr = get_n_params(model_critic)

        print("Training Model:")
        print("Real # model parameters: " + str(n_params_real))
        print("Real # trainable parameters: " + str(n_params_real_tr))
        print("Sim  # model parameters: " + str(n_params_sim))
        print("Sim  # trainable parameters: " + str(n_params_sim_tr))
        print("Critic  # model parameters: " + str(n_params_c))
        print("Critic  # trainable parameters: " + str(n_params_c_tr))

        # Share those modules that are to be shared between real and sim models
        if not self.params.get("disable_domain_weight_sharing"):
            print("Sharing weights between sim and real modules")
            model_real.steal_cross_domain_modules(model_sim)
        else:
            print("NOT Sharing weights between sim and real modules")

        self.model_real = model_real
        self.model_sim = model_sim
        self.model_critic = model_critic
        self.model_oracle_critic = model_oracle_critic
        if self.model_oracle_critic:
            print("Using oracle critic")

        if self.optimizer == "adam":
            Optim = optim.Adam
        elif self.optimizer == "sgd":
            Optim = optim.SGD
        else:
            raise ValueError(f"Unsuppored optimizer {self.optimizer}")

        self.optim_models = Optim(self.model_real.both_domain_parameters(self.model_sim), self.lr, weight_decay=self.weight_decay)
        self.optim_critic = Optim(self.critic_parameters(), self.lr, weight_decay=self.weight_decay)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)

    def set_dataset_names(self, sim_datasets, real_datasets):
        self.sim_dataset_names = sim_datasets
        self.real_dataset_names = real_datasets

    def set_start_epoch(self, epoch):
        self.train_epoch_num = epoch
        self.test_epoch_num = epoch

    def critic_parameters(self):
        for p in self.get_model_parameters(self.model_critic):
            yield p
        if self.model_oracle_critic:
            for p in self.get_model_parameters(self.model_oracle_critic):
                yield p

    def get_model_parameters(self, model):
        params_out = []
        skipped_params = 0
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
        print(str(skipped_params) + " parameters frozen")
        return params_out

    def get_state(self):
        state = {}
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def train_epoch(self, env_list_common=None, env_list_sim=None, data_list_real=None, data_list_sim=None, eval=False):

        if eval:
            self.model_real.eval()
            self.model_sim.eval()
            self.model_critic.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model_real.train()
            self.model_sim.train()
            self.model_critic.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1

        dataset_real_common = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="real", dataset_names=self.real_dataset_names, dataset_prefix="supervised", eval=eval)
        dataset_sim_common = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval)
        dataset_real_halfway = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="real", dataset_names=self.real_dataset_names, dataset_prefix="supervised", eval=eval, halfway_only=True)
        dataset_sim_halfway = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval, halfway_only=True)
        dataset_sim = self.model_sim.get_dataset(data=data_list_sim, envs=env_list_sim, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval)

        print("Beginning supervised epoch:")
        print("   Sim dataset names: ", self.sim_dataset_names)
        print("   Dataset sizes: ")
        print("   dataset_real_common  ", len(dataset_real_common))
        print("   dataset_sim_common  ", len(dataset_sim_common))
        print("   dataset_real_halfway  ", len(dataset_real_halfway))
        print("   dataset_sim_halfway  ", len(dataset_sim_halfway))
        print("   dataset_sim  ", len(dataset_sim))
        print("   env_list_sim_len ", len(env_list_sim))
        print("   env_list_common_len ", len(env_list_common))
        if len(dataset_sim) == 0 or len(dataset_sim_common) == 0:
            print("Missing data! Waiting for RL to generate it?")
            return 0

        dual_model_loader = DualDataloader(
            dataset_a=dataset_real_common,
            dataset_b=dataset_sim_common,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.model_common_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="max"
        )

        sim_loader = DataLoader(
            dataset=dataset_sim,
            collate_fn=dataset_sim.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.model_sim_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False 
        )
        sim_iterator = iter(sim_loader)

        dual_critic_loader = DualDataloader(
            dataset_a=dataset_real_halfway,
            dataset_b=dataset_sim_halfway,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.critic_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="infinite"
        )
        dual_critic_iterator = iter(dual_critic_loader)

        wloss_before_updates_writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{self.run_name}/discriminator_before_updates")
        wloss_after_updates_writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{self.run_name}/discriminator_after_updates")

        samples_real = len(dataset_real_common)
        samples_common = len(dataset_sim_common)
        samples_sim = len(dataset_sim)
        if samples_real == 0 or samples_sim == 0 or samples_common == 0:
            print (f"DATASET HAS NO DATA: REAL: {samples_real > 0}, SIM: {samples_sim > 0}, COMMON: {samples_common}")
            return -1.0

        num_batches = len(dual_model_loader)

        epoch_loss = 0
        count = 0
        critic_elapsed_iterations = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        # Alternate training critic and model
        for real_batch, sim_batch in dual_model_loader:
            if real_batch is None or sim_batch is None:
                print("none")
                continue

            prof.tick("load_model_data")
            # Train the model for model_steps in a row, then train the critic, and repeat
            critic_batch_num = 0

            if count % self.model_steps == 0 and not eval and not self.disable_wloss:
                #print("\nTraining critic\n")
                # Train the critic for self.critic_steps steps
                for cstep in range(self.critic_steps):
                    # Each batch is actually a single rollout (we batch the rollout data across the sequence)
                    # To collect a batch of rollouts, we need to keep iterating
                    real_store = KeyTensorStore()
                    sim_store = KeyTensorStore()
                    for b in range(self.critic_batch_size):
                        # Get the next non-None batch
                        real_c_batch, sim_c_batch = None, None
                        while real_c_batch is None or sim_c_batch is None:
                            real_c_batch, sim_c_batch = next(dual_critic_iterator)
                        prof.tick("critic_load_data")
                        # When training the critic, we don't backprop into the model, so we don't need gradients here
                        with torch.no_grad():
                            images = sim_c_batch["images"][0]
                            seq_len = len(images)
                            instructions = sim_c_batch["instr"][0][:seq_len]
                            
                            #Convert from tokenized list to text-instructions
                            textual_instructions = ""
                            for i in range(len(instructions[0])-1):
                                textual_instructions += self.token2word[int(instructions[0][i])]
                                textual_instructions += " "
                            
                            #Forward to GPT3 
                            object_list = []
                            for chunks in self.nlp(textual_instructions).noun_chunks:
                                prompt = "Is" + chunks + "an object?"
                                answer = self.gpt3(prompt)
                                if answer == "Yes":
                                    object_list.append(chunks)
                                    
                            #Write to csv
                            with open('.csv','a') as f_object:
                                writer_object = csv.writer(f_object)
                                writer_object.writerow([textual_instructions, object_list])
                                f_object.close()
                                

      
def extract_objects():
    P.initialize_experiment()

    setup = P.get_current_parameters()["Setup"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]

    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
    model_real, _ = load_model(setup["model"], setup["real_model_file"], domain="real")
    model_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])

    if P.get_current_parameters()["Training"].get("use_oracle_critic", False):
        model_oracle_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])
        # This changes the name in the summary writer to get a different color plot
        oname = model_oracle_critic.model_name
        model_oracle_critic.set_model_name(oname + "_oracle")
        model_oracle_critic.model_name = oname
    else:
        model_oracle_critic = None

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    # Bidata means that we treat Lani++ and LaniOriginal examples differently, only computing domain-adversarial stuff on Lani++
    bidata = P.get_current_parameters()["Training"].get("bidata", False)
    if bidata == "v2":
        trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
        train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
        train_envs_sim = train_envs
        dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
        dev_envs_sim = dev_envs
    elif bidata:
        trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
        train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
        train_envs_sim = [e for e in train_envs if e < 6000]
        dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
        dev_envs_sim = [e for e in dev_envs if e < 6000]
    else:
        trainer = TrainerBidomain(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)

    print("Beginning training...")
    best_test_loss = 1000
    for epoch in range(1):
        if bidata:
            trainer.train_epoch(env_list_common=train_envs_common, env_list_sim=train_envs_sim, eval=False)
            trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True)
        else:
            trainer.train_epoch(env_list=train_envs, eval=False)
            trainer.train_epoch(env_list=dev_envs, eval=True)

        # if test_loss < best_test_loss:
        #     best_test_loss = test_loss
        #     save_pytorch_model(model_real, real_filename)
        #     save_pytorch_model(model_sim, sim_filename)
        #     save_pytorch_model(model_critic, critic_filename)
        #     print(f"Saved models in: \n Real: {real_filename} \n Sim: {sim_filename} \n Critic: {critic_filename}")

        # print ("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        # save_pytorch_model(model_real, f"tmp/{real_filename}_epoch_{epoch}")
        # save_pytorch_model(model_sim, f"tmp/{sim_filename}_epoch_{epoch}")
        # save_pytorch_model(model_critic, f"tmp/{critic_filename}_epoch_{epoch}")


if __name__ == "__main__":
    extract_objects()