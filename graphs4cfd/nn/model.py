import contextlib
import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from typing import Union, Optional, List, Callable

from ..graph import Graph
from ..loader import DataLoader


class TrainConfig():
    r"""Class to store the training configuration of a model.
    
    Args:
        name (str): Name of the model.
        folder (str, optional): Folder where the model is saved. Defaults to `'./'`.
        checkpoint (Union[None, str], optional): Path of a previous checkpoint to load. If `None`, no checkpoint is loaded. Defaults to `None`.
        tensor_board (Union[None, str], optional): Path of a tensor board to save the training progress. If `None`, no tensor board is saved. Defaults to `None`.
        chk_interval (int, optional): Number of epochs between checkpoints. Defaults to `1`.
        training_loss (Callable, optional): Training loss function. Defaults to `None`.
        validation_loss (Callable, optional): Validation loss function. Defaults to `None`.
        epochs (int, optional): Number of epochs to train. Defaults to `1`.
        num_steps (Union[int, List[int]], optional): Number of rollout steps for each training sample. If `int`, the same number of steps is used during all the training,
            if `List[int]`, a different number of steps is used as the training or validation loss fall velow a tolerance. Defaults to `[1]`.
        add_steps (dict, optional): Dictionary with the parameters of the tolerance and loss function used to add steps. Defaults to `{'tolerance': 0, 'loss':'training'}`.
        batch_size (int, optional): Batch size. Defaults to `1`.
        lr (float, optional): Initial learning rate. Defaults to `1e-3`.
        grad_clip (Union[None, dict], optional): Dictionary with the parameters of the gradient clipping. If `None`, no gradient clipping is used.#
            The dictioary must contain the keys `'epoch'` and `'limit'`, indicating from which epoch the gradient clipping is applied and the maximum gradient norm, respectively.
            Defaults to `None`.
        scheduler (Union[None, dict], optional): Dictionary with the parameters of the learning rate scheduler. The dictioary must contain the keys `'factor'`, `'patience'` and `'loss'`.
            The `'factor'` is the factor by which the learning rate is reduced, `'patience'` is the number of epochs with no improvement after which learning rate will be reduced and `'loss'`
            is the loss function used to monitor the improvement (`'training'` or `'validation'`). Defaults to `None`.
        stopping (float, optional): Minimum value of the learning rate. If the learning rate falls below this value, the training is stopped. Defaults to `0.`.
        mixed_precision (bool, optional): If `True`, mixed precision is used. Defaults to `False`.
        device (Optional[torch.device], optional): Device where the model is trained. If `None`, the model is trained on its current device. Defaults to `None`.
    """

    def __init__(self,
                 name: str,
                 folder: str = './',    
                 checkpoint: Union[None, str] = None,
                 tensor_board: Union[None, str] = None,
                 chk_interval: int = 1,
                 training_loss: Callable = None,
                 validation_loss: Callable = None,
                 epochs: int = 1,
                 num_steps: Union[int, List[int]] = [1],
                 add_steps: dict = {'tolerance': 0, 'loss':'training'},  
                 batch_size: int = 1,
                 lr: float = 1e-3,
                 grad_clip: Union[None, dict] = None,
                 scheduler: Union[None, dict] = None,
                 stopping: float = 0.,
                 mixed_precision: bool = False,
                 device: Optional[torch.device] = None):
        self.name = name    
        self.folder = folder
        self.checkpoint = checkpoint
        self.tensor_board = tensor_board
        self.chk_interval = chk_interval
        self.training_loss = training_loss
        self.validation_loss =validation_loss
        self.epochs = epochs
        self.num_steps = num_steps
        self.add_steps = add_steps
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.stopping = stopping
        self.mixed_precision = mixed_precision
        self.device = device

    def __repr__(self):
        return repr(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__.get(key)


class GNN(nn.Module):
    r"""Base class for all the GNN models.

    Args:
        arch (Optional[Union[None,dict]], optional): Dictionary with the model architecture. Defaults to `None`.
        weights (Optional[Union[None,str]], optional): Path of the weights file. Defaults to `None`.
        model (Optional[Union[None,str]], optional): Path of the checkpoint file. Defaults to `None`.
        device (Optional[torch.device], optional): Device where the model is loaded. Defaults to `torch.device('cpu')`.
    """

    def __init__(self,
                 arch: Optional[Union[None,dict]] = None,
                 weights: Optional[Union[None,str]] = None,
                 checkpoint: Optional[Union[None,str]] = None,
                 device: Optional[torch.device] = torch.device('cpu')):
        """Base class for all the models.
        
        Args:
            arch (Optional[Union[None,dict]], optional): Dictionary with the model architecture. Defaults to `None`.
            weights (Optional[Union[None,str]], optional): Path of the weights file. Defaults to `None`.
            model (Optional[Union[None,str]], optional): Path of the checkpoint file. Defaults to `None`.
            device (Optional[torch.device], optional): Device where the model is loaded. Defaults to `torch.device('cpu')`.
        """
        super().__init__()
        self.device = device
        self.load_model(arch, weights, checkpoint)

    def load_model(self, arch, weights, checkpoint):
        """Loads the model architecture from a arch dictionary and its weights from a weights file, or loads the model from a checkpoint file."""
        if arch is not None and checkpoint is None:
            self.load_arch(arch)
            # To device
            self.to(self.device)
            if weights is not None:
                self.load_state_dict(torch.load(weights, map_location=self.device))
            # Number of output fields
            self.num_fields = arch["decoder"][1][-1] if 'decoder' in arch.keys() else None
        elif arch is None and weights is None and checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location=self.device)
            self.load_arch(checkpoint['arch'])
            # To device
            self.to(self.device)
            self.load_state_dict(checkpoint['weights'])
            # Number of output fields
            self.num_fields = checkpoint['arch']["decoder"][1][-1] if 'decoder' in checkpoint['arch'].keys() else None
        return


    # To be overwritten
    def load_arch(self, arch: dict):
        """Defines the hyper-parameters of the model. It must be overloaded by each model instancing the `graphs4cfd.models.Model` class.

        Args:
            arch (dict): Dictionary with the architecture of the model. Its structure depends on the model.
        """
        pass
    
    # To be overwritten
    def forward(self, graph: Graph, t: Optional[int] = None):
        """Forwrad pass (or time step) of the model. It must be overloaded by each model instancating the `graphs4cfd.models.Model` class.

        Args:
            graph (Graph): Graph object with the input data.
            t (int): current time-point
        """
        pass
 
    def fit(self,
            train_config: TrainConfig,
            train_loader: DataLoader,
            val_loader: Union[None, DataLoader] = None):
        """Trains the model.
        
        Args:
            config (TrainConfig): Configuration of the training.
            train_loader (DataLoader): Training data loader.
            test_loader (Union[None, DataLoader], optional): Test data loader. If `None`, no validation is performed. Defaults to `None`.        
        """
        # Change the training device if needed
        if train_config['device'] is not None and train_config['device'] != self.device:
            self.to(train_config['device'])
            self.device = train_config['device']
        # Set the training loss
        criterion = train_config['training_loss']
        max_n_out = train_config['num_steps'][-1] # Maximun number of predicted time-steps
        num_steps = iter(train_config['num_steps'])
        n_out = next(num_steps)
        # Load checkpoint
        checkpoint = None
        scheduler  = None
        if train_config['checkpoint'] is not None and os.path.exists(train_config['checkpoint']):
            print("Training from an existing check-point:", train_config['checkpoint'])
            checkpoint = torch.load(train_config['checkpoint'], map_location=self.device)
            self.load_state_dict(checkpoint['weights'])
            optimiser = torch.optim.Adam(self.parameters(), lr=checkpoint['lr'])
            optimiser.load_state_dict(checkpoint['optimiser'])
            if train_config['scheduler'] is not None: 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=train_config['scheduler']['factor'], patience=train_config['scheduler']['patience'], eps=0.)
                scheduler.load_state_dict(checkpoint['scheduler'])
            while n_out < checkpoint['n_out']: n_out = next(num_steps)
            initial_epoch = checkpoint['epoch']+1
        # Initialise optimiser and scheduler if not previous check-point is used
        else:
            # If a .chk is given but it does not exist such file, notify the user
            if train_config['checkpoint'] is not None:
                print("Not matching check-point file:", train_config['checkpoint'])
            print('Training from randomly initialised weights')
            optimiser = optim.Adam(self.parameters(), lr=train_config['lr'])
            if train_config['scheduler'] is not None: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=train_config['scheduler']['factor'], patience=train_config['scheduler']['patience'], eps=0.)
            initial_epoch = 1
        # If .chk to save exists rename the old version to .bck
        path = os.path.join(train_config["folder"], train_config["name"]+".chk")
        if os.path.exists(path):
            print('Renaming', path, 'to:', path+'.bck')
            os.rename(path, path+'.bck')
        # Initialise tensor board writer
        if train_config['tensor_board'] is not None: writer = SummaryWriter(os.path.join(train_config["tensor_board"], train_config["name"]))
        # Initialise automatic mixed-precision training
        scaler = None
        if train_config['mixed_precision']:
            print("Training with automatic mixed-precision")
            scaler = torch.cuda.amp.GradScaler()
            # Load previos scaler
            if checkpoint is not None and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
        # Print before training
        print(f'Training on device: {self.device}')
        print(f'Number of trainable parameters: {self.num_params}')
        # Training loop
        for epoch in tqdm(range(initial_epoch,train_config['epochs']+1), desc="Completed epochs", leave=False, position=0):
            if optimiser.param_groups[0]['lr'] < train_config['stopping']:
                print(f"The learning rate is smaller than {train_config['stopping']}. Stopping training.")
                self.save_checkpoint(path, n_out, epoch, optimiser, scheduler=scheduler, scaler=scaler)
                break
            print("\n")
            print(f"Hyperparameters: n_out = {n_out}, lr = {optimiser.param_groups[0]['lr']}")
            self.train()
            training_loss = 0.
            gradients_norm = 0.
            for iteration, data in enumerate(train_loader):
                data = data.to(self.device)
                for t in range(n_out):
                    if t > 0:
                        data.field = self.shift_and_replace(data.field, pred.detach())
                    with torch.cuda.amp.autocast() if train_config['mixed_precision'] else contextlib.nullcontext(): # Use automatic mixed-precision
                        pred = self.forward(data, t)
                        loss = criterion(data, pred, data.target[:,self.num_fields*t:self.num_fields*(t+1)])
                    # Back-propagation
                    if train_config['mixed_precision']:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # Save training loss and gradients norm before applying gradient clipping to the weights
                    training_loss  += loss.item()/n_out
                    gradients_norm += self.grad_norm2()/n_out
                    # Update the weights
                    if train_config['mixed_precision']:
                        # Clip the gradients
                        if train_config['grad_clip'] is not None and epoch > train_config['grad_clip']["epoch"]:
                            scaler.unscale_(optimiser)
                            nn.utils.clip_grad_norm_(self.parameters(), train_config['grad_clip']["limit"])
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        # Clip the gradients
                        if train_config['grad_clip'] is not None and epoch > train_config['grad_clip']["epoch"]:
                            nn.utils.clip_grad_norm_(self.parameters(), train_config['grad_clip']["limit"])
                        optimiser.step()
                    # Reset the gradients
                    optimiser.zero_grad()
            training_loss  /= (iteration+1)
            gradients_norm /= (iteration+1)
            # Display on terminal
            print(f"Epoch: {epoch:4d}, Training   loss: {training_loss:.4e}, Gradients: {gradients_norm:.4e}")
            # Testing
            if val_loader is not None:
                validation_criterion = train_config['validation_loss']
                self.eval()
                with torch.no_grad(): 
                    validation_loss = 0.
                    for iteration, data in enumerate(val_loader):
                        data = data.to(self.device)
                        for t in range(max_n_out):
                            if t > 0:
                                data.field = self.shift_and_replace(data.field, pred)
                            pred = self.forward(data, t)
                            validation_loss += validation_criterion(data, pred, data.target[:,self.num_fields*t:self.num_fields*(t+1)]).item()/max_n_out
                    validation_loss /= (iteration+1)
                    print(f"Epoch: {epoch:4d}, Validation loss: {validation_loss:.4e}")
            # Log in TensorBoard
            if train_config['tensor_board'] is not None:
                writer.add_scalar('Loss/train', training_loss,   epoch)
                if val_loader: writer.add_scalar('Loss/test',  validation_loss, epoch)
            # Update lr
            if train_config['scheduler']['loss'][:2] == 'tr':
                scheduler_loss = training_loss 
            elif train_config['scheduler']['loss'][:3] == 'val':
                scheduler_loss = validation_loss 
            scheduler.step(scheduler_loss)
            # Create training checkpoint
            if not epoch%train_config["chk_interval"]:
                print('Saving check-point in:', path)
                self.save_checkpoint(path, n_out, epoch, optimiser, scheduler=scheduler, scaler=scaler)
            # Encrease n_out for next epoch
            if train_config['add_steps']['loss'][:2] == 'tr':
                tolerance_loss = training_loss 
            elif train_config['add_steps']['loss'][:3] == 'val':
                tolerance_loss = validation_loss 
            else:
                raise NameError("Invalid parameter config['add_steps']['loss].")
            if (tolerance_loss < train_config['add_steps']['tolerance'] and n_out < max_n_out):
                n_out = next(num_steps)
                optimiser = optim.Adam(self.parameters(), lr=train_config["lr"])
                if train_config['scheduler']["patience"] is not None: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=train_config['scheduler']["factor"], patience=train_config['scheduler']["patience"], eps=0.)
        writer.close()
        print("Finished training")
        return

    def solve(self, graph: Graph, n_out: int) -> torch.Tensor:
        """Evaluate the model on the graph for n_out time-steps."""
        assert n_out > 0, "n_out must be greater than 0."
        self.eval()
        with torch.no_grad():
            if type(graph) is list:
                graph = Batch.from_data_list(graph)
            else:
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
            graph.to(self.device)
            field   = graph.field
            outputs = torch.zeros( (graph.num_nodes, self.num_fields*n_out), device=self.device)
            # Time steps
            for t in range(n_out):
                pred = self.forward(graph, t)
                outputs[:,self.num_fields*t:self.num_fields*(t+1)] = pred
                # Re-feed or restore data.field
                graph.field = self.shift_and_replace(graph.field, pred) if (t+1 < n_out) else field
            return outputs

    def shift_and_replace(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Shift the fields in x by num_fields and replace the last num_fields with y."""
        x = torch.roll(x, -self.num_fields, dims=1)
        x[:, -self.num_fields:] = y
        return x

    def save_checkpoint(self,
                        file_name: str,
                        n_out: int,
                        epoch: int,
                        optimiser: torch.optim.Optimizer,
                        scheduler: Union[None, dict] = None,
                        scaler: Union[None, dict] = None):
        """Saves the model parameters, the optimiser state and the current value of the training hyper-parameters.
        The saved file can be used to resume training with the `graphs4cfd.nn.model.Model.fit` method."""
        checkpoint = {
            'arch'     : self.arch,
            'weights'  : self.state_dict(),
            'optimiser': optimiser.state_dict(),
            'n_out'    : n_out,
            'lr'       : optimiser.param_groups[0]['lr'],
            'epoch'    : epoch, 
        }
        if scheduler is not None: checkpoint['scheduler'] = scheduler.state_dict()
        if scaler    is not None: checkpoint['scaler']    = scaler.state_dict()
        torch.save(checkpoint, file_name)
        return

    @property
    def num_params(self):
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def grad_norm2(self):
        """Returns the L2 norm of the gradients."""
        norm = 0.
        for p in self.parameters():
            if p.requires_grad:
                norm += p.grad.data.norm(2).item()**2
        return norm**.5
