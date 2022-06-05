# TODO move this into the metamodel
import torch
import torchdiffeq


class RecVectorField(torch.nn.Module):
    def __init__(self, X, func):
        """Defines a controlled vector field.

        Arguments:
            X: As cdeint.
            func: As cdeint.
        """
        super(RecVectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.X = X
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_x = self.X(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        out = self.func(control_x, z)
        # just like in ContinuousRNNConverter,
        # without (-1,1) constraints, we get nan with GRU...
        out = torch.tanh(out)
        # out = out.clamp(-1, 1)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        # out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out


def recodeint(X, z0, func, t, adjoint=True, **kwargs):
    r"""Solves a system of recurrent ordinary differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s, x_s)ds
    ```
    where z is a tensor of any shape, and x is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """

    control_x = X(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_x.shape[:-1] != z0.shape[:-1]:
        raise ValueError("X did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_x.shape), tuple(control_x.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if control_x.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = RecVectorField(X, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out
