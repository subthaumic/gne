from sympy import Function
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

import numpy as np
import scipy

import timeit
from tqdm import tqdm

from gne.data.dataset import SimplicialDataset, collate_unique_vertices
from gne.utils.geometries import Euclidean, Geometry
from gne.utils.complex import Complex
from gne.utils.geometry_to_complex import geometric_weights, kneighbors_complex
from gne.utils.optim import RiemannianSGD
from gne.utils.loss import L2, Loss
from gne.visualization import visualize
from gne.models.Config import Config


class GeometricEmbedding(torch.nn.Module):
    def __init__(
        self,
        source_geometry: Geometry = None,
        source_complex: Complex = None,
        target_complex: Complex = None,
        target_geometry: Geometry = None,
        config: Config = None,
        weight_function: Function = geometric_weights,
        loss_function: Loss = None,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = ReduceLROnPlateau,
    ):
        """
        Initialize a GeometricEmbedding object with specified source and target
        geometries, complexes, training configurations, weight function, loss function,
        optimizer, and learning rate scheduler.

        Runs various (non-exhaustive) validation checks on input and initialized class.

        Parameters:
            source_geometry (Geometry): The geometry of the source, equipped with
                a data torch.tensor in source_geometry.sample. This is the data that
                will be embedded.
            source_complex (Complex, optional): Intermediate simplicial complex
                associated to source.
            target_complex (Complex, optional): Intermediate simplicial complex
                associated to target.
            target_geometry (Geometry): The geometry of the target, defaults to
                2d Euclidean space
            config (Config): Training configurations.
            weight_function (Function): Function to calculate weights of simplicial
                complex, defaults to "geometric weights" as discussed in article.
            loss_function (Loss): Loss function used for optimization, defaults to
                L2-norm.
            optimizer (Optimizer): Optimizer used for parameter updates, defaults to
                Riemannian SGD and standard SGD in case of Euclidean target space.
            scheduler (LRScheduler): Scheduler used to adjust the learning rate,
                defaults to torch's ReduceLROnPlateau.
        """
        self._validate_inputs(source_geometry, source_complex, target_geometry, config)

        super().__init__()
        target_geometry, config = self._set_mutable_defaults(target_geometry, config)
        self._set_instance_variables(
            source_geometry,
            source_complex,
            target_complex,
            target_geometry,
            config,
            weight_function,
            loss_function,
            optimizer,
            scheduler,
        )
        self._validate_initialization()
        self.__repr__()

    def __repr__(self):
        out = "gne.GeometricEmbedding(\n"
        out += f"\t source_geometry={self.source_geometry},\n"
        out += f"\t source_complex={self.source_complex},\n"
        out += f"\t target_complex={self.target_complex},\n"
        out += f"\t target_geometry={self.target_geometry},\n"
        out += f"\t weight_function={self.weight_function},\n"
        out += f"\t loss_function={self.loss_function},\n"
        out += f"\t config={self.config}\n"
        out += ")"
        return out

    def _validate_inputs(
        self, source_geometry, source_complex, target_geometry, config
    ):
        # Validate source_geometry
        if source_geometry is None:
            raise ValueError("source_geometry must be provided.")
        elif not isinstance(source_geometry, Geometry):
            raise ValueError("source_geometry must be a valid instance of Geometry.")
        elif source_geometry.sample is None:
            raise ValueError("source_geometry must have a sample tensor.")

        # Validate source complex if provided
        if source_complex is not None and not isinstance(source_complex, Complex):
            raise ValueError("source_complex must be a valid instance of Complex.")

        # Validate target_geometry if provided
        if target_geometry is not None and not isinstance(target_geometry, Geometry):
            raise ValueError(
                "target_geometry must be a valid Geometry instance or None."
            )

        # Validate config if provided
        if config is not None and not isinstance(config, Config):
            raise ValueError("config must be an instance of Config or None.")

    def _set_mutable_defaults(self, target_geometry, config):
        if config is None:
            config = Config()
        if target_geometry is None:
            target_geometry = Euclidean(dimension=config.target_geometry["dimension"])
        else:
            config.target_geometry["dimension"] = target_geometry.dimension
        return target_geometry, config

    def _set_instance_variables(
        self,
        source_geometry,
        source_complex,
        target_complex,
        target_geometry,
        config,
        weight_function,
        loss_function,
        optimizer,
        scheduler,
    ):
        # set configurations
        self.config = config

        # set source and target geometries
        self.source_geometry = source_geometry
        self.source_complex = source_complex
        self.target_complex = target_complex
        self.target_geometry = target_geometry
        self._initialize_target_sample(
            method=self.config.target_geometry["initialization_method"]
        )

        if self.config.training["batch_size"] == -1:
            # batch_size == -1 -> auto-set batch_size
            max_num_simplices = scipy.special.binom(
                self.source_geometry.sample_size + 1,
                self.config.source_complex["k_neighbours"],
            )
            self.config.training["batch_size"] = min(
                512, int(np.ceil(max_num_simplices / 100))
            )
            # NOTE: Not sure why 512 was chosen, but probably has to do with
            # efficient computation. Might want to exchange for something more sensible.
        elif (
            self.config.training["batch_size"] == 0
            or self.config.training["batch_size"] == torch.inf
        ):
            # batch_size == 0 -> turn off batch processing
            # (by setting batch_size to maximal possible number of simplices)
            max_num_simplices = scipy.special.binom(
                self.source_geometry.sample_size + 1,
                self.config.source_complex["k_neighbours"],
            )
            self.config.training["batch_size"] = int(2 * max_num_simplices)

        # TODO: make choice of weight_function
        self.weight_function = weight_function

        # TODO: make choice of loss_function part of Config
        # TODO: combine choice of loss_function and Lagrange multipliers
        #   e.g. { "dim 0": [L2(), multiplier], "dim 1": ...}
        # NOTE: Currently use L2([a,b,...]) = a*L2 + b*L2 + ... to handle dimensions
        if loss_function is None:
            self.loss_function = L2(self.config.loss["lagrange_multipliers"])
        else:
            self.loss_function = loss_function

        # set optimizer and scheduler
        # TODO: make choice of optimizer and scheduler part of Config
        self.optimizer = self._create_optimizer(optimizer)

        self.scheduler = scheduler(
            self.optimizer,
            factor=self.config.scheduler["factor"],
            patience=self.config.scheduler["patience"],
            cooldown=self.config.scheduler["cooldown"],
        )

    def _initialize_target_sample(self, method):
        # Initialize a latent representation of source in target_geometry
        # if not already provided
        if self.target_geometry.sample is None:
            if method == "PCA":
                _, _, V = torch.pca_lowrank(self.source_geometry.sample)
                principal_directions = V[:, : self.target_geometry.dimension]
                principal_components = torch.matmul(
                    self.source_geometry.sample, principal_directions
                )
                self.target_geometry.sample = principal_components
            elif method == "UMAP":
                from umap import UMAP

                umap_model = UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=self.target_geometry.dimension,
                )
                umap_embedding = umap_model.fit_transform(self.source_geometry.sample)
                self.target_geometry.sample = torch.tensor(umap_embedding)
            elif method == "random":
                max_dist = self.source_geometry.compute_distances().max()
                self.target_geometry.sample = torch.rand(
                    self.source_geometry.sample_size,
                    self.target_geometry.dimension,
                    dtype=torch.float64,
                ).uniform_(-0.5 * max_dist, 0.5 * max_dist)
            else:
                raise ValueError(
                    f"""
                        Initialization method {method} not implemented.
                        Initialize target_geometry.sample yourself.
                    """
                )

            self.target_geometry.sample.requires_grad_()
            self.target_geometry.sample_size = self.target_geometry.sample.size()[0]

        # TODO: provide gpu functionality (over batches) ?
        # if self.config.training['cuda']:
        #     self.target_geometry.sample.cuda()

        # ensure target_geometry sample requires gradients in any case
        if not self.target_geometry.sample.requires_grad:
            self.target_geometry.sample.requires_grad_()

    def _validate_initialization(self):
        # ensure sample sizes in source and target coincide
        if not self.source_geometry.sample_size == self.target_geometry.sample_size:
            raise ValueError(
                "source_geometry and target_geometry have different sample_size"
            )
        # TODO: Are there other checks, tests, and validations I should run?

    def _create_optimizer(self, optimizer_class=None):
        """
        Creates an optimizer instance based on the geometry of the target. If a specific
        optimizer class is provided, it uses that; otherwise, it defaults to SGD if the
        target geometry is Euclidean and to Riemannian SGD if not.

        Args:
            optimizer_class (Optional[Type[torch.optim.Optimizer]]): The class of the
                optimizer to use. If None, selects torch.optim.SGD for Euclidean
                targets and RiemannianSGD otherwise.

        Returns:
            torch.optim.Optimizer: An instance of the specified or determined optimizer
            class.
        """
        params = [self.target_geometry.sample]
        lr = self.config.training["learning_rate"]

        # Build default keyword arguments
        kwargs = {"lr": lr}
        if not isinstance(self.target_geometry, Euclidean):
            kwargs["metric"] = self.target_geometry.metric

        # Set default optimizer based on target_geometry
        if optimizer_class is None:
            if isinstance(self.target_geometry, Euclidean):
                optimizer_class = torch.optim.SGD
            else:
                optimizer_class = RiemannianSGD

        return optimizer_class(params, **kwargs)

    def forward(
        self,
        fout=None,
        pout=None,
        plot_loss=None,
    ):
        ## Initialization
        t_start = timeit.default_timer()

        if fout is None:
            fout = self.config.output["interim_data_path"]
        if pout is None:
            pout = self.config.output["interim_data_path"]
        if plot_loss is None:
            plot_loss = self.config.output["plot_loss"]

        # Compute simplicial complexes from source and target geometry
        if self.source_complex is None:
            self.source_complex = Complex.creator(
                creator_func=kneighbors_complex,
                geometry=self.source_geometry,
                k_neighbours=self.config.source_complex["k_neighbours"],
                max_dim=self.config.source_complex["max_dim"],
                weight_fn=self.weight_function,
            )

        if self.target_complex is None:
            self.target_complex = Complex.creator(
                creator_func=kneighbors_complex,
                geometry=self.target_geometry,
                k_neighbours=self.config.target_complex["k_neighbours"],
                max_dim=self.config.target_complex["max_dim"],
                weight_fn=self.weight_function,
            )

        # Set up dataset of simplices from source complex (which is immutable)
        source_simplices = SimplicialDataset(self.source_complex)
        print(
            "n_batches ~= "
            + f"{np.ceil(2*len(source_simplices)/self.config.training['batch_size'])}"
        )

        ## Optimization
        pbar = tqdm(range(self.config.training["epochs"]), ncols=180)
        sequence_of_epoch_losses = []
        deltas = []
        earlystop_count = 0

        for epoch in pbar:
            epoch_loss = 0

            # Set up dataset of simplicies from target complex for the current epoch
            target_simplices = SimplicialDataset(self.target_complex)
            # combine with immutable source_simplices from above
            combined_simplices = ConcatDataset([source_simplices, target_simplices])

            # prepare current epoch's data for batch processing
            dataloader = DataLoader(
                dataset=combined_simplices,
                collate_fn=collate_unique_vertices,
                batch_size=self.config.training["batch_size"],
                shuffle=True,
            )

            # run batchwise optimization
            for set_of_vertex_ids in dataloader:
                # Clear any existing gradients
                self.optimizer.zero_grad()

                # Compute subcomplex of source_complex on current batch of vertices
                source_complex = self.source_complex(set_of_vertex_ids)

                # Compute pushforward of source_complex
                pushforward_complex = source_complex.copy(include_weights=False)
                for simplex in pushforward_complex:
                    indices = [int(v) for v in simplex.vertices]
                    pushforward_weight = self.weight_function(
                        self.target_geometry(indices)
                    )
                    pushforward_complex.update_weight(simplex, pushforward_weight)

                # get weights of source_complex and pushforward_complex
                source_weights = source_complex.get_weights()
                pushforward_weights = pushforward_complex.get_weights()

                # Compute subcomplex of target_complex on current batch of vertices
                target_complex = self.target_complex(set_of_vertex_ids)

                # # Compute pullback of target_complex
                # pullback_complex = target_complex.copy(include_weights=False)
                # for simplex in pullback_complex:
                #     pullback_weight = self.source_complex.get_weight(simplex)
                #     if pullback_weight is None:
                #         target_weight = self.target_complex.get_weight(simplex)
                #         if target_weight is not None:
                #             pullback_weight = 1.1 * target_weight.detach()
                #         # NOTE: As long as pullback_weight is larger than
                #           target_weight, Lp loss yields a repulsive gradient descent.
                #         # This behaviour has not been tested for other loss functions.
                #     pullback_complex.update_weight(simplex, pullback_weight)

                # NOTE: It would be way more logical/symmetric to just calculate
                #   the missing weight!
                # Compute pullback of target_complex
                pullback_complex = target_complex.copy(include_weights=False)
                for simplex in pullback_complex:
                    pullback_weight = self.source_complex.get_weight(simplex)
                    if pullback_weight is None:
                        indices = [int(v) for v in simplex.vertices]
                        pullback_weight = self.weight_function(
                            self.target_geometry(indices)
                        )
                    pullback_complex.update_weight(simplex, pullback_weight)
                # TODO: Run tests to see if this helps or hinders.

                # get weights of target_complex and pullback_complex
                target_weights = target_complex.get_weights()
                pullback_weights = pullback_complex.get_weights()

                # Calculate loss via symmetric comparison
                # ( source <-> pushforward ) + ( pullback <-> target )
                pushforward_loss = self.loss_function(
                    source_weights, pushforward_weights
                )

                pullback_loss = self.loss_function(target_weights, pullback_weights)

                loss = pushforward_loss + pullback_loss
                epoch_loss += loss.item()

                # backprop
                loss.backward()

                self.optimizer.step()

                # Propagate optimization step to target_complex, i.e. recompute
                # weighted simplicial complex associated to current target geometry
                # TODO: I'd like to update self.target_complex more efficiently
                # TODO: Take into account that updating the full complex may remove or
                #       add edges.
                #       Is this something I want?
                #       Currently this can happen; and since batches go through
                #       "original" complex
                #       of given epoch, I need to check that loss != 0, otherwise can
                #       get vanishing gradients.
                self.target_complex = Complex.creator(
                    creator_func=kneighbors_complex,
                    geometry=self.target_geometry,
                    k_neighbours=self.config.target_complex["k_neighbours"],
                    max_dim=self.config.target_complex["max_dim"],
                    weight_fn=self.weight_function,
                )

            # Update learning rate
            self.scheduler.step(epoch_loss)

            # Evaluate early-stop criterions
            sequence_of_epoch_losses.append(epoch_loss)

            if len(sequence_of_epoch_losses) >= 2:
                deltas.append(
                    np.abs(sequence_of_epoch_losses[-1] - sequence_of_epoch_losses[-2])
                )
            else:
                deltas.append(torch.inf)

            delta_threshold = np.quantile(deltas, self.config.earlystop["quantile"])

            earlystop_criteria = [
                epoch > self.config.training["burnin"],
                self.scheduler._last_lr[0] <= self.config.earlystop["lr_threshold"],
                deltas[-1] <= delta_threshold,
            ]

            if all(earlystop_criteria):
                earlystop_count += 1
                if earlystop_count >= self.config.earlystop["patience"]:
                    print(f"\nStopped at epoch {epoch}")
                    break

            # Update progress bar
            pbar.set_description(
                f"loss: {epoch_loss}, "
                + f"delta: {deltas[-1]}, "
                + f"threshold: {delta_threshold}, "
                + f"lr: {self.scheduler._last_lr[0]}, "
                + "earlystop_count:"
                + f"{earlystop_count} / {self.config.earlystop['patience']}"
            )

        ## Post-Optimization Wrap-up
        embedding = self.target_geometry.sample.detach()
        t_stop = timeit.default_timer()

        duration = t_stop - t_start
        print(f"duration: {(duration):.2f} sec = {duration/60:.2f} min")

        # save embedding to file
        if fout is not None:
            np.savetxt(fout + ".csv", embedding, delimiter=",")

        # plot loss
        if plot_loss:
            visualize.plot_training(
                sequence_of_epoch_losses, title_name="Loss", file_name=pout
            )

        return embedding
