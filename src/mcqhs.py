"""
mcqhs.py — Simulación Monte Carlo NVT para fluido de esferas duras cuántico
===========================================================================
Traducción completa de mcqhs.f (Fortran 77) a Python con Numba.

Potencial de par Mie(50,49) con corrección cuántica de Wigner-Kirkwood
a primer orden (expansión semiclásica en lambda_B).

Referencia del potencial:
    Jover et al., J. Chem. Phys. 137, 144505 (2012)

Corrección cuántica:
    Wigner, E. P., Phys. Rev. 40, 749 (1932)
    Kirkwood, J. G., Phys. Rev. 44, 31 (1933)

Ecuación de estado teórica de referencia:
    Serna & Gil-Villegas, Mol. Phys. (2016)

Archivos de entrada:
    mc.in   — parámetros de simulación
    mc.old  — configuración previa (si NRUN=1)

Archivos de salida:
    mc.dat  — resultados principales (energía, CV, Z)
    uav.dat — series temporales de energía y CV
    rdf.dat — función de distribución radial g(r)
    mc.new  — configuración final de posiciones

Uso:
    python mcqhs.py

Dependencias:
    numpy >= 1.21
    numba >= 0.55

febrero 2024 — traducción a Python
===========================================================================
"""

import math
import sys
import numpy as np
from numba import njit

# ===========================================================================
# CONSTANTES GLOBALES (equivalente a mc.inc)
# ===========================================================================
NPART = 864    # Número máximo de partículas
NACC  = 20     # Número de acumuladores estadísticos
NG    = 1000   # Número de bins del histograma de g(r)


# ===========================================================================
# FUNCIONES NUMBA — NÚCLEO COMPUTACIONAL
# Todas estas funciones son JIT-compiladas con @njit para máxima velocidad.
# Equivalen al Fortran original en rendimiento numérico.
# ===========================================================================

@njit(cache=True)
def _imagen_minima(dx, dy, dz, Y2, Z2, YC, ZC):
    """
    Convención de imagen mínima en 3D.
    Devuelve el vector separación más corto considerando periodicidad.

    Equivalente Fortran:
        IF (X.GT.0.5) X=X-1.0
        ELSE IF (X.LT.-0.5) X=X+1.0
        ...
    """
    if   dx >  0.5: dx -= 1.0
    elif dx < -0.5: dx += 1.0
    if   dy >  Y2:  dy -= YC
    elif dy < -Y2:  dy += YC
    if   dz >  Z2:  dz -= ZC
    elif dz < -Z2:  dz += ZC
    return dx, dy, dz


@njit(cache=True)
def _potencial_par(rr, SS, XMIN2, CTE, CWK, UWT0):
    """
    Evalúa el potencial Mie(50,49) + corrección Wigner-Kirkwood.

    Parámetros
    ----------
    rr    : distancia al cuadrado (unidades de caja)
    SS    : sigma^2 en unidades de caja
    XMIN2 : radio de corte al cuadrado
    CTE   : prefactor Mie = 50*(50/49)^49
    CWK   : prefactor WK  = lambda_B^2 / (24*pi)
    UWT0  : desplazamiento WCA (valor del pot. en el mínimo)

    Retorna
    -------
    uhs : contribución clásica Mie/HS
    uwk : corrección cuántica Wigner-Kirkwood

    Potencial:
        u_HS(r) = CTE * [(s/r)^50 - (s/r)^49] + 1
        u_WK(r) = CTE*CWK * [2450*(s/r)^52 - 2352*(s/r)^51] - UWT0 - 1
        (cero si r >= XMIN)

    Nota sobre las potencias: se calculan por multiplicaciones
    sucesivas para eficiencia, igual que en el Fortran original.
    """
    if rr >= XMIN2:
        return 0.0, 0.0

    SR2  = SS / rr
    SR   = math.sqrt(SR2)          # (sigma/r)^1
    SR10 = SR2*SR2*SR2*SR2*SR2     # (sigma/r)^10
    SR20 = SR10*SR10               # (sigma/r)^20
    SR50 = SR20*SR20*SR10          # (sigma/r)^50
    SR49 = SR20*SR20*SR2*SR2*SR2*SR2*SR  # (sigma/r)^49
    SR51 = SR50*SR                 # (sigma/r)^51
    SR52 = SR50*SR2                # (sigma/r)^52

    uhs = CTE*(SR50 - SR49) + 1.0
    uwk = CTE*CWK*(2450.0*SR52 - 2352.0*SR51) - UWT0 - 1.0
    return uhs, uwk


@njit(cache=True)
def _energia_particula(i, xi, yi, zi,
                       RX, RY, RZ, N,
                       SS, XMIN2, CTE, CWK, UWT0,
                       Y2, Z2, YC, ZC):
    """
    Energía de la partícula i en posición (xi,yi,zi) con todas las demás.
    Complejidad O(N).

    Equivalente Fortran: bucle DO 2 J=1,N en SUBROUTINE MONTECARLO.
    """
    uhs_sum = 0.0
    uwk_sum = 0.0
    for j in range(N):
        if j == i:
            continue
        dx = RX[j] - xi
        dy = RY[j] - yi
        dz = RZ[j] - zi
        dx, dy, dz = _imagen_minima(dx, dy, dz, Y2, Z2, YC, ZC)
        rr = dx*dx + dy*dy + dz*dz
        uhs, uwk = _potencial_par(rr, SS, XMIN2, CTE, CWK, UWT0)
        uhs_sum += uhs
        uwk_sum += uwk
    return uhs_sum, uwk_sum


@njit(cache=True)
def _energia_total(RX, RY, RZ, N,
                   SS, XMIN2, CTE, CWK, UWT0,
                   Y2, Z2, YC, ZC):
    """
    Energía potencial total del sistema — bucle doble O(N^2).
    Solo pares únicos (i < j).

    Equivalente Fortran: SUBROUTINE ENERGIA(UTOTHS, UTOTWK).
    Se llama una sola vez al inicio; durante el MC se actualiza
    de forma incremental.

    Retorna
    -------
    UTOTHS : energía total clásica (parte Mie/HS)
    UTOTWK : energía total corrección cuántica WK
    """
    UTOTHS = 0.0
    UTOTWK = 0.0
    for i in range(N - 1):
        rxi = RX[i]; ryi = RY[i]; rzi = RZ[i]
        for j in range(i + 1, N):
            dx = RX[j] - rxi
            dy = RY[j] - ryi
            dz = RZ[j] - rzi
            dx, dy, dz = _imagen_minima(dx, dy, dz, Y2, Z2, YC, ZC)
            rr = dx*dx + dy*dy + dz*dz
            uhs, uwk = _potencial_par(rr, SS, XMIN2, CTE, CWK, UWT0)
            UTOTHS += uhs
            UTOTWK += uwk
    return UTOTHS, UTOTWK


@njit(cache=True)
def _gofr_update(RX, RY, RZ, N, G, S, XHIST,
                 Y2, Z2, YC, ZC, NG_size):
    """
    Actualiza el histograma de pares G[L] para g(r).
    Se llama cada NFDR pasos MC.

    Fórmula del bin:
        L = int((r - 0.75*sigma) * XHIST / (1 - 0.75*sigma))

    El origen del histograma está en r = 0.75*sigma para capturar
    la región de contacto del potencial Mie.

    Equivalente Fortran: SUBROUTINE GOFR.
    """
    RMAX = 0.5 * 0.5  # radio máximo = mitad de la caja
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = RX[i] - RX[j]
            dy = RY[i] - RY[j]
            dz = RZ[i] - RZ[j]
            dx, dy, dz = _imagen_minima(dx, dy, dz, Y2, Z2, YC, ZC)
            rr = dx*dx + dy*dy + dz*dz
            if rr > RMAX:
                continue
            r = math.sqrt(rr)
            L = int((r - 0.75*S) * XHIST / (1.0 - 0.75*S))
            if 0 <= L < NG_size:
                G[L] += 1.0


@njit(cache=True)
def _ciclo_metropolis(RX, RY, RZ, N, NMOVE, NSUB0, NFDR,
                      DISPL, SS, XMIN2, CTE, CWK, UWT0,
                      Y2, Z2, YC, ZC, XN, TEMP,
                      XHIST, S, G, NG_size,
                      UTOTHS_init, UTOTWK_init):
    """
    Ciclo principal de Metropolis NVT.

    Por cada paso:
      1. Selecciona partícula aleatoria
      2. Propone desplazamiento en [-DISPL/2, +DISPL/2]
      3. Aplica condiciones periódicas de frontera
      4. Calcula ΔU = U(nueva) - U(vieja)
      5. Acepta si ΔU ≤ 0, o con probabilidad exp(-ΔU/T_sim)
         donde T_sim = 1.5 (fijo para imitar esferas duras con Mie(50,49))
      6. Acumula estadísticas cada NSUB pasos

    Equivalente Fortran: SUBROUTINE MONTECARLO (etiquetas 1-6).

    Retorna arrays con las series temporales de los observables.
    """
    UTOTHS = UTOTHS_init
    UTOTWK = UTOTWK_init
    UTOT   = UTOTHS + UTOTWK

    USUBAVHS = 0.0
    USUBAVWK = 0.0
    NACCPT   = 0
    NSUB     = NSUB0
    NFDR0    = NFDR

    # Acumuladores estadísticos (equivalen al array ACC del Fortran)
    acc1  = 0.0   # ACC(1)  : contador de pasos
    acc2  = 0.0   # ACC(2)  : suma de UTOTHS
    acc12 = 0.0   # ACC(12) : suma de UTOTWK
    acc5  = 0.0   # ACC(5)  : suma de UTOT^2
    acc10 = 0.0   # ACC(10) : suma de UAV (para std)
    acc18 = 0.0   # ACC(18) : suma de UAV^2
    acc14 = 0.0   # ACC(14) : suma de CVAV
    acc16 = 0.0   # ACC(16) : suma de CVAV^2
    n_gofr = 0    # ACC(3)  : muestras de g(r)

    # Arrays de salida (series temporales de subpromedios)
    max_nave   = NMOVE // NSUB0 + 2
    out_ncount = np.zeros(max_nave, dtype=np.int64)
    out_accept = np.zeros(max_nave, dtype=np.float64)
    out_uavhs  = np.zeros(max_nave, dtype=np.float64)
    out_uavwk  = np.zeros(max_nave, dtype=np.float64)
    out_uav    = np.zeros(max_nave, dtype=np.float64)
    out_std1   = np.zeros(max_nave, dtype=np.float64)
    out_cvav   = np.zeros(max_nave, dtype=np.float64)
    out_std2   = np.zeros(max_nave, dtype=np.float64)

    NAVE = 0

    for NCOUNT in range(1, NMOVE + 1):

        # ── Selección aleatoria de partícula ──────────────────────────────
        i = int(np.random.random() * N)

        rxi = RX[i]; ryi = RY[i]; rzi = RZ[i]

        # ── Propuesta de movimiento ───────────────────────────────────────
        xnew = rxi + DISPL * (np.random.random() - 0.5)
        ynew = ryi + DISPL * (np.random.random() - 0.5)
        znew = rzi + DISPL * (np.random.random() - 0.5)

        # Condiciones periódicas de frontera (caja 0..1 en x, 0..YC en y, z)
        if   xnew > 1.0: xnew -= 1.0
        elif xnew < 0.0: xnew += 1.0
        if   ynew > YC:  ynew -= YC
        elif ynew < 0.0: ynew += YC
        if   znew > ZC:  znew -= ZC
        elif znew < 0.0: znew += ZC

        # ── Energía de la partícula en posición nueva y vieja ─────────────
        uhs_new, uwk_new = _energia_particula(
            i, xnew, ynew, znew,
            RX, RY, RZ, N,
            SS, XMIN2, CTE, CWK, UWT0, Y2, Z2, YC, ZC)

        uhs_old, uwk_old = _energia_particula(
            i, rxi, ryi, rzi,
            RX, RY, RZ, N,
            SS, XMIN2, CTE, CWK, UWT0, Y2, Z2, YC, ZC)

        dEnHS = uhs_new - uhs_old
        dEnWK = uwk_new - uwk_old
        dEn   = dEnHS + dEnWK

        # ── Criterio de Metropolis ────────────────────────────────────────
        # T_sim = 1.5 (fijo por construcción para que Mie(50,49) ≈ HS)
        # Para usar la temperatura del input: reemplazar 1.5 por TEMP
        if dEn <= 0.0:
            acepta = True
        else:
            acepta = np.random.random() < math.exp(-dEn / 1.5)

        if acepta:
            RX[i]   = xnew
            RY[i]   = ynew
            RZ[i]   = znew
            UTOTHS += dEnHS
            UTOTWK += dEnWK
            UTOT    = UTOTHS + UTOTWK
            NACCPT += 1

        # ── Acumulación de estadísticas ───────────────────────────────────
        acc1  += 1.0
        acc2  += UTOTHS
        acc12 += UTOTWK
        acc5  += UTOT * UTOT
        USUBAVHS += UTOTHS
        USUBAVWK += UTOTWK

        # ── Muestreo de g(r) ─────────────────────────────────────────────
        if NCOUNT == NFDR0:
            _gofr_update(RX, RY, RZ, N, G, S, XHIST,
                         Y2, Z2, YC, ZC, NG_size)
            n_gofr += 1
            NFDR0  += NFDR

        # ── Escritura de subpromedios ─────────────────────────────────────
        if NCOUNT >= NSUB or NCOUNT == NMOVE:
            NAVE    += 1
            xnave    = float(NAVE)
            xncount  = float(NCOUNT)

            UAVHS = USUBAVHS / (XN * xncount)
            UAVWK = USUBAVWK / (XN * xncount)
            UAV   = UAVHS + UAVWK
            UAV2  = acc5 / xncount
            CVAV  = (UAV2 - XN*XN*UAV*UAV) / (TEMP * TEMP * XN)

            acc10 += UAV
            acc18 += UAV * UAV
            acc14 += CVAV
            acc16 += CVAV * CVAV

            xp1  = acc10 / xnave;  xp2 = acc18 / xnave
            xp3  = acc14 / xnave;  xp4 = acc16 / xnave
            std1 = math.sqrt(max(0.0, xp2 - xp1*xp1))
            std2 = math.sqrt(max(0.0, xp4 - xp3*xp3))

            idx = NAVE - 1
            out_ncount[idx] = NCOUNT
            out_accept[idx] = NACCPT / xncount
            out_uavhs [idx] = UAVHS
            out_uavwk [idx] = UAVWK
            out_uav   [idx] = UAV
            out_std1  [idx] = std1
            out_cvav  [idx] = CVAV
            out_std2  [idx] = std2

            NSUB += NSUB0

    return (NAVE, n_gofr,
            out_ncount[:NAVE], out_accept[:NAVE],
            out_uavhs[:NAVE],  out_uavwk[:NAVE], out_uav[:NAVE],
            out_std1[:NAVE],   out_cvav[:NAVE],  out_std2[:NAVE])


# ===========================================================================
# FUNCIONES PYTHON — INICIALIZACIÓN Y E/S
# ===========================================================================

def leer_parametros(filename='mc.in'):
    """
    Lee los parámetros de simulación desde mc.in.

    Equivalente Fortran: bloque READ en SUBROUTINE COMIENZO.

    Formato de mc.in:
        N
        RHO  XLB
        NRUN NMOVE NSUB
        DISPL
        NFDR LFDR XHIST
        NAC NCX NCY NCZ
        XA(1) YA(1) ZA(1)
        ... (NAC líneas)
    """
    with open(filename, 'r') as f:
        lineas = [l.strip() for l in f if l.strip()]

    N                   = int(lineas[0])
    RHO, XLB            = map(float, lineas[1].split())
    tok                 = lineas[2].split()
    NRUN, NMOVE, NSUB   = int(tok[0]), int(tok[1]), int(tok[2])
    DISPL               = float(lineas[3])
    tok                 = lineas[4].split()
    NFDR                = int(tok[0])
    LFDR                = tok[1].upper() in ('.TRUE.', 'TRUE', '1', 'T')
    XHIST               = float(tok[2])
    tok                 = lineas[5].split()
    NAC, NCX, NCY, NCZ  = int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3])

    XA = np.zeros(NAC); YA = np.zeros(NAC); ZA = np.zeros(NAC)
    for k in range(NAC):
        tok       = lineas[6 + k].split()
        XA[k], YA[k], ZA[k] = float(tok[0]), float(tok[1]), float(tok[2])

    return (N, RHO, XLB, NRUN, NMOVE, NSUB, DISPL,
            NFDR, LFDR, XHIST, NAC, NCX, NCY, NCZ, XA, YA, ZA)


def inicializar(params):
    """
    Calcula todos los parámetros derivados y genera la configuración inicial.

    Equivalente Fortran: SUBROUTINE COMIENZO.

    Retorna un diccionario con todas las constantes del sistema
    y los arrays de posiciones RX, RY, RZ.
    """
    (N, RHO, XLB, NRUN, NMOVE, NSUB, DISPL,
     NFDR, LFDR, XHIST, NAC, NCX, NCY, NCZ, XA, YA, ZA) = params

    pi   = math.acos(-1.0)
    XLB2 = XLB * XLB
    XLB3 = XLB2 * XLB

    # Posición del mínimo del potencial efectivo (ajuste polinomial en lambda_B)
    XMIN = (1.0194 + 0.14983*XLB - 0.18698*XLB2 + 0.078628*XLB3)

    # Temperatura física reducida: T* = kT/ε = 1/(2π·λ_B²)
    TEMP = 0.5 / (XLB2 * pi)

    XN   = float(N)
    XNCX = float(NCX); XNCY = float(NCY); XNCZ = float(NCZ)

    # Relaciones de aspecto (XC=1 por construcción)
    XC = 1.0
    YC = XNCY / XNCX
    ZC = XNCZ / XNCX
    Y2 = YC / 2.0
    Z2 = ZC / 2.0

    # Longitud de caja en unidades de sigma, de rho = N/(Lx³·YC·ZC)
    XL = (XN / (RHO * YC * ZC))**(1.0/3.0)
    YL = XL * YC
    ZL = XL * ZC

    # sigma en unidades de la caja
    S  = 1.0 / XL
    SS = S * S

    # Constantes del potencial Mie(50,49) + WK
    XMIN2 = XMIN * XMIN * SS                    # radio de corte^2 (unid. caja)
    CTE   = 50.0 * (50.0/49.0)**49              # prefactor Mie
    CWK   = XLB2 / (24.0 * pi)                  # prefactor WK
    UX0   = CTE * (XMIN**(-50.0) - XMIN**(-49.0))
    UWK0  = 50.0*49.0*XMIN**(-52.0) - 49.0*48.0*XMIN**(-51.0)
    UWT0  = CTE * CWK * UWK0 + UX0              # desplazamiento WCA

    # Reescala desplazamiento a unidades de la caja
    DISPL_sc = DISPL * S

    # ── Configuración inicial ─────────────────────────────────────────────
    RX = np.zeros(N); RY = np.zeros(N); RZ = np.zeros(N)

    if NRUN == 0:
        # Genera red cúbica replicando la celda unitaria
        idx = 0
        for IZ in range(1, NCZ+1):
            for J in range(NAC):
                for IX in range(1, NCX+1):
                    for IY in range(1, NCY+1):
                        RX[idx] = ((IX - 1.0) + XA[J]) / XNCX
                        RY[idx] = ((IY - 1.0) + YA[J]) * YC / XNCY
                        RZ[idx] = ((IZ - 1.0) + ZA[J]) * ZC / XNCZ
                        idx += 1
    else:
        # Lee configuración previa de mc.old
        datos = np.loadtxt('mc.old')
        RX[:] = datos[:N, 0]
        RY[:] = datos[:N, 1]
        RZ[:] = datos[:N, 2]

    estado = dict(
        N=N, RHO=RHO, XLB=XLB, XLB2=XLB2, XLB3=XLB3,
        NRUN=NRUN, NMOVE=NMOVE, NSUB=NSUB, DISPL=DISPL_sc,
        NFDR=NFDR, LFDR=LFDR, XHIST=XHIST,
        NAC=NAC, NCX=NCX, NCY=NCY, NCZ=NCZ,
        pi=pi, XN=XN, TEMP=TEMP, XMIN=XMIN,
        XC=XC, YC=YC, ZC=ZC, Y2=Y2, Z2=Z2,
        XL=XL, YL=YL, ZL=ZL, S=S, SS=SS,
        XMIN2=XMIN2, CTE=CTE, CWK=CWK, UWT0=UWT0,
        RX=RX, RY=RY, RZ=RZ,
        G=np.zeros(NG)
    )
    return estado


def energia_teorica(estado):
    """
    Energía de referencia por la ecuación de estado de Serna & Gil-Villegas (2016).
    No se usa en el ciclo MC; es solo comparación teórica.

    Equivalente Fortran: bloque de ec. de estado en SUBROUTINE MONTECARLO.
    """
    RHO  = estado['RHO'];  XLB  = estado['XLB']
    XLB2 = estado['XLB2']; TEMP = estado['TEMP']; pi = estado['pi']

    et  = RHO * pi / 6.0
    et2 = et * et
    de1 = 1.6593854484;  de2 = -1.0927115150;  de3 = -1.1188233921

    etq    = (1.0 + de1*XLB)*et + (de2*XLB + de3*XLB2)*et2
    etq2   = etq*etq;  etq3 = etq2*etq
    detq   = 1.0 - etq;  detq2 = detq*detq;  detq3 = detq2*detq
    zqhs   = (1.0 + etq + etq2 - etq3) / detq3 - 1.0
    dxletq = de1*et + (de2 + 2.0*de3*XLB)*et2
    UQHST  = (zqhs/etq) * dxletq * math.sqrt(TEMP)
    UQHST  = 0.5 * UQHST / math.sqrt(2.0*pi)
    return UQHST


def imprimir_cabecera(estado, out=None):
    """
    Imprime los parámetros iniciales en pantalla y en mc.dat.
    Equivalente Fortran: bloque de WRITEs en SUBROUTINE COMIENZO.
    """
    streams = [sys.stdout]
    if out is not None:
        streams.append(out)

    def w(s):
        for f in streams:
            print(s, file=f)

    w("*" * 55)
    w("***  SIMULACION MONTE CARLO NVT QHS (MIE(50,49)-WK)  ***")
    w("*" * 55)
    w("")
    w(f"  N= {estado['N']:6d}  RHO= {estado['RHO']:.8f}  TEMP= {estado['TEMP']:.5f}"
      f"  XLB= {estado['XLB']:.5f}  RC= {estado['XMIN']:.8f}")
    w("")
    w(f"  NRUN= {estado['NRUN']:4d}  NMOVE= {estado['NMOVE']:10d}  NSUB= {estado['NSUB']:9d}")
    w("")
    w(f"  NAC,NCX,NCY,NCZ= {estado['NAC']:4d}{estado['NCX']:4d}"
      f"{estado['NCY']:4d}{estado['NCZ']:4d}")
    w("")
    w(f"  DISPL= {estado['DISPL']:.8f}")
    w("")
    if estado['NRUN'] == 0:
        w("  INICIO DE UNA RED")
    else:
        w("  INICIO DE CONFIGURACION PREVIA")
    w("")
    w(f"  TAMANO DE CAJA (unid. programa): XC={estado['XC']:.5f} "
      f"YC={estado['YC']:.5f} ZC={estado['ZC']:.5f}")
    w(f"  TAMANO DE CAJA (unid. sigma):    XL={estado['XL']:.5f} "
      f"YL={estado['YL']:.5f} ZL={estado['ZL']:.5f}")
    w(f"  SIGMA EN UNIDADES DE LA CAJA:    {estado['S']:.8f}")
    w("")
    w("  Primeras 20 posiciones:")
    for i in range(min(20, estado['N'])):
        w(f"    {estado['RX'][i]:.8f}  {estado['RY'][i]:.8f}  {estado['RZ'][i]:.8f}")
    w("")


def ejecutar_montecarlo(estado, out_mc=None, out_uav=None):
    """
    Llama al ciclo de Metropolis compilado con Numba y escribe resultados.

    Equivalente Fortran: SUBROUTINE MONTECARLO.
    """
    # Cálculo de energía inicial
    UTOTHS, UTOTWK = _energia_total(
        estado['RX'], estado['RY'], estado['RZ'], estado['N'],
        estado['SS'], estado['XMIN2'], estado['CTE'], estado['CWK'],
        estado['UWT0'], estado['Y2'], estado['Z2'], estado['YC'], estado['ZC'])

    UPERP = (UTOTHS + UTOTWK) / estado['XN']
    UQHST = energia_teorica(estado)

    streams_mc = [sys.stdout]
    if out_mc: streams_mc.append(out_mc)

    def wmc(s):
        for f in streams_mc: print(s, file=f)

    wmc(f"\n  inicio de corrida")
    wmc(f"  energia inicial  {UPERP:.8g}\n")

    # ── Ciclo Metropolis (JIT compilado) ──────────────────────────────────
    np.random.seed(123456789)  # equivalente a ISEED=-123456789 de Fortran
    (NAVE, n_gofr,
     nc, ac, uhs, uwk, uav, std1, cv, std2) = _ciclo_metropolis(
        estado['RX'], estado['RY'], estado['RZ'],
        estado['N'], estado['NMOVE'], estado['NSUB'], estado['NFDR'],
        estado['DISPL'], estado['SS'], estado['XMIN2'],
        estado['CTE'], estado['CWK'], estado['UWT0'],
        estado['Y2'], estado['Z2'], estado['YC'], estado['ZC'],
        estado['XN'], estado['TEMP'],
        estado['XHIST'], estado['S'], estado['G'], NG,
        UTOTHS, UTOTWK)

    estado['G_nsamples'] = n_gofr   # ACC(3): muestras de g(r)

    # Escribe series temporales
    hdr = ("  NMOV           AC      UHS     UWK     UAV    std1"
           "    UTEO      CV    std2")
    wmc(hdr)
    for k in range(NAVE):
        linea = (f"  {nc[k]:12d}  {ac[k]:8.4f}  {uhs[k]:8.4f}  {uwk[k]:8.4f}"
                 f"  {uav[k]:8.4f}  {std1[k]:8.4f}  {UQHST:8.4f}"
                 f"  {cv[k]:8.4f}  {std2[k]:8.4f}")
        wmc(linea)
        if out_uav:
            print(f"  {nc[k]:12d}  {uhs[k]:10.4f}  {uwk[k]:10.4f}"
                  f"  {uav[k]:10.4f}  {std1[k]:10.4f}  {cv[k]:10.4f}"
                  f"  {std2[k]:10.4f}", file=out_uav)

    wmc(f"\n  Num total de configuraciones desde inicio  {float(nc[-1]):12.0f}")
    wmc(f"  Energia promedio desde el inicio  {uav[-1]:10.4f}\n")


def calcular_radial(estado, out_mc=None):
    """
    Normaliza el histograma G[L] para obtener g(r) y calcula Z.

    Equivalente Fortran: SUBROUTINE RADIAL.
    """
    XNAV = float(estado['G_nsamples'])
    if XNAV == 0.0:
        print("  [RADIAL] No hay muestras de g(r), se omite.")
        return

    XL    = estado['XL']; YL = estado['YL']; ZL = estado['ZL']
    S     = estado['S'];  XN = estado['XN']; pi = estado['pi']
    RHO   = estado['RHO']; XHIST = estado['XHIST']
    XC    = estado['XC'];  XLB2  = estado['XLB2']
    CWK   = estado['CWK']; CTE   = estado['CTE']
    G     = estado['G']

    DELR  = (1.0 - 0.75*S) * XL / XHIST
    VOLME = XL * YL * ZL
    CONST = (2.0/3.0) * XN*XN * pi * XNAV / VOLME
    NEND  = int((0.5*XC - 0.75*S) * XHIST / (1.0 - 0.75*S))

    scol = 0.0; fr = 0.0; xlcol = 0.0; fa = 0.0
    r_arr = []; gr_arr = []

    for L in range(1, NEND+1):
        X   = float(L)
        R   = (1.0 - 0.75*S)*(2.0*X - 1.0)*XL / (XHIST*2.0) + 0.75
        RB  = (1.0 - 0.75*S)*(X - 1.0)*XL / XHIST + 0.75
        RSQ = R*R; RS3 = R*R*R
        RS5 = RSQ*RS3; RS10 = RS5*RS5
        RS20 = RS10*RS10; RS40 = RS20*RS20
        RS50 = RS40*RS10; RS51 = RS50*R
        RS52 = RS50*RSQ; RS53 = RS50*RS3
        RBCU = RB*RB*RB
        RC   = RB + DELR
        RC3  = RC*RC*RC
        GR   = G[L-1] / (CONST*(RC3 - RBCU))

        DU1   = 49.0/RS50 - 50.0/RS51
        DU2   = 392.0*CWK*(306.0/RS52 - 325.0/RS53)
        DU    = CTE*(DU1 + DU2)
        SCOL3 = GR*DU*RS3*DELR
        SCOL2 = GR*DU*RSQ*DELR

        if DU < 0.0:
            scol += SCOL3;  fr   += SCOL2
        else:
            xlcol += SCOL3; fa   += SCOL2

        r_arr.append(R); gr_arr.append(GR)

    # Escribe rdf.dat
    with open('rdf.dat', 'w') as frdf:
        for r, g in zip(r_arr, gr_arr):
            print(f"  {r:.6f}  {g:.6f}", file=frdf)

    # Parámetros de colisión y factor de compresibilidad Z
    sfin   = scol / fr if fr != 0.0 else 0.0
    if fa > 0.0:
        xlcolfin = xlcol / fa
        freqa    = 24.0 * fa * RHO
    else:
        xlcolfin = 0.0
        freqa    = 0.0
    freqr = -24.0 * fr * RHO
    zeta  = 2.0*pi*(sfin*freqr - xlcolfin*freqa)/3.0 + 1.0

    linea = (f"  NUR  ={freqr:.5f}  NUA  ={freqa:.5f}"
             f"  <s> ={sfin:.5f}  <l>={xlcolfin:.5f}  Z={zeta:.5f}")
    print(linea)
    if out_mc:
        print(linea, file=out_mc)


def guardar_final(estado, out_mc=None):
    """
    Guarda la configuración final en mc.new para poder reanudar.

    Equivalente Fortran: SUBROUTINE FINAL.
    """
    with open('mc.new', 'w') as f:
        for i in range(estado['N']):
            print(f"  {estado['RX'][i]:.8f}  {estado['RY'][i]:.8f}"
                  f"  {estado['RZ'][i]:.8f}", file=f)

    streams = [sys.stdout]
    if out_mc: streams.append(out_mc)
    for f in streams:
        print("\n\n  configuracion final", file=f)
        print(f"  {'RX':>20s}{'RY':>22s}{'RZ':>22s}", file=f)
        for i in range(estado['N']):
            print(f"  {estado['RX'][i]:.8f}  {estado['RY'][i]:.8f}"
                  f"  {estado['RZ'][i]:.8f}", file=f)


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================

def main():
    """
    Punto de entrada principal.

    Equivalente Fortran: program mcqhs — secuencia
        CALL COMIENZO → CALL MONTECARLO → CALL RADIAL → CALL FINAL
    """
    print("Compilando funciones Numba (primera ejecucion puede tardar ~30 s)...")

    # ── 1. Lectura de parámetros ──────────────────────────────────────────
    params = leer_parametros('mc.in')

    # ── 2. Inicialización ─────────────────────────────────────────────────
    estado = inicializar(params)

    # ── 3. Archivos de salida ─────────────────────────────────────────────
    with (open('mc.dat', 'w') as out_mc,
          open('uav.dat', 'w') as out_uav):

        imprimir_cabecera(estado, out_mc)

        # ── 4. Ciclo Monte Carlo ──────────────────────────────────────────
        ejecutar_montecarlo(estado, out_mc, out_uav)

        # ── 5. Función de distribución radial y factor Z ──────────────────
        calcular_radial(estado, out_mc)

        # ── 6. Configuración final ────────────────────────────────────────
        guardar_final(estado, out_mc)

    print("\nSimulacion completada. Archivos generados:")
    print("  mc.dat  — resultados principales")
    print("  uav.dat — series temporales de energía")
    print("  rdf.dat — funcion de distribucion radial g(r)")
    print("  mc.new  — configuracion final (usar como mc.old con NRUN=1)")


if __name__ == '__main__':
    main()
