import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import metatopia as mtp
import metatopia.utils as mtpu
import optax
from common import get_meta_train_iterator_fns
from omegaconf import DictConfig


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="meta_train")
def main(cfg: DictConfig) -> None:
    # Set up the solver and define loss functions
    domain_shape = (cfg.top_opt.Ny, cfg.top_opt.Nx)
    coords = mtpu.get_grid_coords(domain_shape)
    # solver = mtp.solver.setup_solver_cvxopt(**cfg.top_opt)

    # Create a random key for use in the script
    key = jr.PRNGKey(cfg.seed)
    model_key, learner_key = jr.split(key)

    # Set up the design parameterisation
    model = mtp.models.get_model_by_name(cfg.base_model.name)(
        key=model_key, **cfg.base_model.hyperparameters
    )

    params, static = eqx.partition(model, eqx.is_array)

    opt = optax.adam(cfg.meta_learner.inner_lr)
    opt_state = opt.init(params)

    # Loss for a single batch
    def sample_loss(params, static, sample):
        e = sample["strain_energy"].reshape(-1, 1)
        input = jnp.concatenate([coords, e], axis=1)
        model = eqx.combine(params, static)
        output = jax.vmap(model)(input)

        # Calculate mse between output and strain energy
        mse = jnp.mean((output - e) ** 2)

        return mse

    def loss_fn(params, static, batch):
        batch_loss = eqx.filter_vmap(sample_loss, in_axes=(None, None, 0))(
            params, static, batch
        )

        return jnp.mean(batch_loss)

    def train_step(params, static, sample, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, static, sample)

        updates, opt_state = opt.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    train_iterator_fn, _ = get_meta_train_iterator_fns(cfg)
    train_iterator = train_iterator_fn(batch_size=256)

    num_epochs = 1
    num_batches = len(train_iterator)

    train_step = eqx.filter_jit(train_step)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in train_iterator:
            # Update the model
            params, opt_state, loss = train_step(params, static, batch, opt_state)

            epoch_loss += loss

        avg_epoch_loss = epoch_loss / num_batches

        mtp.logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}"
        )

    # Optionally, save the trained model
    pretrained_model = eqx.combine(params, static)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    mtp.logger.info(f"Saving the pretrained model in {output_dir}.")

    mtp.utils.serialise_tree(pretrained_model, "pretrained_model", output_dir)


if __name__ == "__main__":
    main()
