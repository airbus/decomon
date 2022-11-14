from .backward_cloning import NumpyCROWNModel


def clone(
    model,
    convex_domain={},
    ibp=False,
    forward=True,
    method="crown",
    back_bounds=[],
    shared=True,
    dico_grid={},
    final_ibp=True,
    final_forward=True,
    **kwargs,
):

    # use shared in the layers
    if method != "crown":
        raise NotImplementedError()
    if not shared:
        raise NotImplementedError()

    return NumpyCROWNModel(
        model,
        ibp=False,
        forward=True,
        convex_domain=convex_domain,
        has_back_bounds=bool(len(back_bounds)),
        dico_grid=dico_grid,
        **kwargs,
    )
