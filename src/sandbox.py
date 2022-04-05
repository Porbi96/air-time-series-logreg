import skfda
import matplotlib.pyplot as plt
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.preprocessing.smoothing.kernel_smoothers import NadarayaWatsonSmoother
from skfda.ml.classification import LogisticRegression
from src.wav_utils import create_wav_fdatagrid


def plot_fpca(fd, n_components=3, n_basis=7):
    basis = skfda.representation.basis.BSpline(n_basis=n_basis)
    basis_fd = fd.to_basis(basis)
    fpca_discretized = FPCA(n_components=n_components)
    fpca = FPCA(n_components=n_components)

    fpca_discretized.fit(fd)
    fpca.fit(basis_fd)

    fpca_discretized.components_.plot()
    FPCAPlot(
        basis_fd.mean(),
        fpca.components_,
        10,
        fig=plt.figure(figsize=(6, 2*4)),
        n_rows=3,
    ).plot()
    plt.show()


def test():
    directory = '../data/wiertarka/CM_5_zabkow_wiatrak/'
    fd = create_wav_fdatagrid(directory)
    fd.plot(linewidth=0.5)

    # smoother = NadarayaWatsonSmoother()
    # fd_smoothed = smoother.fit_transform(fd)
    # fd_smoothed.plot()
    plt.show()

    plot_fpca(fd)


if __name__ == "__main__":
    test()
