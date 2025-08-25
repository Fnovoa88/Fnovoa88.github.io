"""Utilidades para reclasificar capas predictivas."""

from typing import Sequence

import numpy as np


def compute_youden_threshold(y_pred: Sequence[float],
                             y_true: Sequence[int],
                             nfr: int) -> float:
    """Retorna el corte que maximiza el índice de Youden."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    cortes = np.linspace(0, 1, nfr)
    mejor_corte, mejor_youden = 0.0, -1.0
    for corte in cortes:
        y_bin = y_pred >= corte
        sens = ((y_bin == 1) & (y_true == 1)).mean()
        esp = ((y_bin == 0) & (y_true == 0)).mean()
        youden = sens + esp - 1
        if youden > mejor_youden:
            mejor_corte, mejor_youden = float(corte), youden
    return mejor_corte


def reclasificar_capa(y_pred: Sequence[float],
                       y_true: Sequence[int],
                       radioButtonManual,
                       radioButtonIndiceYouden,
                       spinBoxValorEspecificado,
                       spinBoxNfr):
    """Reclasifica la capa según el método seleccionado."""
    if radioButtonManual.isChecked():
        corte = spinBoxValorEspecificado.value()
    elif radioButtonIndiceYouden.isChecked():
        corte = compute_youden_threshold(y_pred, y_true,
                                         spinBoxNfr.value())
    else:
        raise ValueError("Debe seleccionar un método de corte")
    return (np.asarray(y_pred) >= corte).astype(int)


def actualizar_widgets(radioButtonManual,
                      radioButtonIndiceYouden,
                      spinBoxValorEspecificado,
                      spinBoxNfr) -> None:
    """Habilita los spinBox correspondientes según la opción elegida."""
    spinBoxValorEspecificado.setEnabled(radioButtonManual.isChecked())
    spinBoxNfr.setEnabled(radioButtonIndiceYouden.isChecked())
