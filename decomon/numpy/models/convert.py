from .backward_cloning import NumpyCROWNModel


def clone(
    model,
    convex_domain=None,
    ibp=False,
    forward=True,
    method="crown",
    back_bounds=None,
    shared=True,
    dico_grid=None,
    final_ibp=True,
    final_forward=True,
    **kwargs,
):

    # use shared in the layers
    if convex_domain is None:
        convex_domain = {}
    if back_bounds is None:
        back_bounds = []
    if dico_grid is None:
        dico_grid = {}
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
