from .SRSolver import SRSolver

def create_solver(opt):
    if opt['mode'] == 'ar':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver