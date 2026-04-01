C ===========================================================================
C  mcqhs.f — Simulación Monte Carlo NVT para fluido de esferas duras
C            continuo con potencial Mie(50,49) y corrección cuántica
C            de Wigner-Kirkwood a primer orden.
C
C  Descripción:
C    Simula N partículas esféricas en una caja periódica 3D usando el
C    algoritmo de Metropolis. Calcula energía interna promedio, 
C    función de distribución radial g(r), calor especifico, 
C    parámetros de colisión (<s>, <l>) y 
C    factor de compresibilidad Z.
C
C  Potencial por pares Mie(50,49)(Jover et al., J. Chem. Phys. 137, 144505, 2012):
C    u(r) = CTE * eps * [ (s/r)^50 - (s/r)^49 ] + eps,  si r < r_c
C    u(r) = 0,                                          si r >= r_c
C    CTE  = 50 * (50/49)^49
C    s    = sigma (diámetro de colisión, cero del potencial)
C    eps  = profundidad del mínimo
C    r_c  = sigma
C
C  Corrección cuántica Wigner-Kirkwood (1er orden en beta=1/kT):
C    u_WK(r) = (lambda_B^2 / 24*pi) * nabla^2 u(r)
C    lambda_B = hbar/sqrt(m*kT) es la longitud de onda termica
C    de de Broglie. La expansion es valida en el limite semiclásico
C    (lambda_B << sigma).
C    Temperatura reducida: T* = 1 / (2*pi*lambda_B^2)
C
C  Referencias originales:
C    Wigner, E. P., Phys. Rev. 40, 749 (1932)
C    Kirkwood, J. G., Phys. Rev. 44, 31 (1933); 45, 116 (1934)
C
C  Flujo de ejecución:
C    COMIENZO   -> lee parámetros, genera/lee configuración inicial
C    MONTECARLO -> ciclo de Metropolis (equilibración + producción)
C    RADIAL     -> normaliza g(r) y calcula Z
C    FINAL      -> escribe configuración final para reinicio
C
C  Archivos de entrada:
C    mc.in   -> parámetros de simulación
C    mc.old  -> configuración previa (si NRUN=1)
C
C  Archivos de salida:
C    mc.dat  -> resultados principales (energía, CV, Z)
C    uav.dat -> series temporales de energia y CV
C    rdf.dat -> función de distribución radial g(r)
C    mc.new  -> configuración final de posiciones
C
C  febrero 2024
C ===========================================================================

      program mcqhs

C  Convención de tipos: IMPLICIT DOUBLE PRECISION(A-H,O-Z)
C  Todas las variables cuyos nombres empiecen con A-H o O-Z son
C  DOUBLE PRECISION (64 bits). Las que empiezan con I-N son INTEGER.
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)

C  Variables globales compartidas entre subrutinas (posiciones, 
C  acumuladores, dimensiones de caja, etc.)
      include "mc.inc"

C  pi = arccos(-1), notación d0 indica constante double precision
      pi = dacos(-1.d0)

C  Archivo de salida principal: resultados de la simulación
      OPEN(UNIT=7,FILE='mc.dat',STATUS='unknown')

C  Secuencia principal de ejecución
      CALL COMIENZO
      CALL MONTECARLO
      CALL RADIAL
      CALL FINAL

      STOP
      END

C ===========================================================================
C  SUBROUTINE COMIENZO
C
C  Proposito:
C    Inicializar todos los parámetros del sistema antes del ciclo MC.
C    Lee mc.in, calcula dimensiones de caja en unidades del programa,
C    y genera o lee la configuración inicial de partículas.
C
C  Variables de entrada (leídas de mc.in):
C    N       -> número de partículas (debe cumplir N = NAC*NCX*NCY*NCZ)
C    RHO     -> densidad reducida rho* = N*sigma^3/V
C    XLB     -> longitud de onda termica de de Broglie (lambda_B/sigma)
C    NRUN    -> 0: iniciar de red  |  1: continuar de mc.old
C    NMOVE   -> número total de intentos de movimiento
C    NSUB    -> frecuencia de escritura de subpromedios
C    DISPL   -> desplazamiento máximo en unidades de sigma
C    NFDR    -> frecuencia de muestreo de g(r) (cada cuantos pasos)
C    LFDR    -> .TRUE./.FALSE. activa/desactiva el calculo de g(r)
C    XHIST   -> número de bins del histograma de g(r)
C    NAC     -> número de partículas por celda unitaria
C    NCX,NCY,NCZ -> replicas de la celda en x, y, z
C    XA,YA,ZA    -> coordenadas fraccionarias de la base de la celda
C ===========================================================================
      SUBROUTINE COMIENZO
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"
      DIMENSION XA(6),YA(6),ZA(6)

C  --- Inicialización de acumuladores ---
C  ACC es el array de acumuladores estadísticos (energía, CV, g(r), etc.)
C  Se pone a cero antes de comenzar cualquier calculo.
      DO 3 I=1,NACC
      ACC(I)=0.0D00
    3 CONTINUE

C  --- Lectura de parámetros desde mc.in ---
      OPEN (UNIT=1,FILE='mc.in',STATUS='OLD')
      READ(1,*) N
      READ(1,*) RHO,XLB
      READ(1,*) NRUN,NMOVE,NSUB
      READ(1,*) DISPL
      READ(1,*) NFDR, LFDR, XHIST
      READ(1,*) NAC,NCX,NCY,NCZ
      READ(1,*)(XA(I),YA(I),ZA(I),I=1,NAC)

C  --- Calculo de parametros derivados de XLB ---
C  XLB2, XLB3: potencias de la longitud de onda térmica
      XLB2 = XLB*XLB
      XLB3 = XLB2*XLB

C  Posición del mínimo del potencial efectivo u(r) + u_WK(r)
C  como función de lambda_B.
C  Ajuste polinomial obtenido numericamente:
C    XMIN = 1.0194 + 0.14983*XLB - 0.18698*XLB^2 + 0.078628*XLB^3
      XMIN = 1.0194d0 + 0.14983d0*XLB -0.18698d0*XLB2
      XMIN = XMIN + 0.078628d0*XLB3

C  Temperatura física del sistema cuántico, determinada por XLB:
C  Temperatura fisica reducida T* = kT/epsilon = 1 / (2*pi*lambda_B^2)
      TEMP = 0.5d0/XLB2/pi
C  Caracteriza el grado de cuanticidad del fluido (no es la temperatura
C  de simulación del potencial, son magnitudes independientes)

C  --- Escritura de parámetros de entrada (pantalla y mc.dat) ---
      write(6,100)
      write(7,100)
      write(6,101) N,RHO,TEMP,XLB,XMIN
      write(7,101) N,RHO,TEMP,XLB,XMIN
      write(6,1011) NRUN,NMOVE,NSUB
      write(7,1011) NRUN,NMOVE,NSUB
      write(6,1012) NAC,NCX,NCY,NCZ
      write(7,1012) NAC,NCX,NCY,NCZ
      write(6,1013) DISPL
      write(7,1013) DISPL
      IF (NRUN.EQ.0.0D00) WRITE(6,102)
      IF (NRUN.EQ.0.0D00) WRITE(7,102)
      IF (NRUN.NE.0.0D00) WRITE(6,103)
      IF (NRUN.NE.0.0D00) WRITE(7,103)

C  --- Conversion a unidades del programa ---
C  En el programa todas las distancias se escalan con Lx (longitud de
C  la caja en x), de modo que las coordenadas van de 0 a 1 en x.
C  Esto simplifica las condiciones periódicas de frontera.
      XN=DFLOAT(N)
      XNCX=DFLOAT(NCX)
      XNCY=DFLOAT(NCY)
      XNCZ=DFLOAT(NCZ)

C  Relaciones de aspecto de la caja (XC=1 siempre por construcción)
      XC=1.0D00
      YC=XNCY/XNCX
      ZC=XNCZ/XNCX

C  Mitades de las dimensiones de caja (usadas en la imagen mínima)
      Y2=YC/2.0D00
      Z2=ZC/2.0D00

C  Lx en unidades de sigma, calculado de la densidad:
C    rho = N / (Lx * Ly * Lz) = N / (Lx^3 * YC * ZC)
C    -> Lx = (N / (rho * YC * ZC))^(1/3)
      XL=(XN/(RHO*YC*ZC))**(1.0D00/3.0D00)
      YL=XL*YC
      ZL=XL*ZC

C  Sigma en unidades de la caja: S = sigma/Lx = 1/Lx
      S=1.0D00/XL
      SS=S*S

C  --- Constantes del potencial ---
C  XMIN2: radio de corte al cuadrado en unidades de la caja.
      XMIN2 = XMIN*XMIN*SS

C  CTE: prefactor del potencial Mie(50,49) = 50*(50/49)^49
      CTE = 50.0D00*((50.0D00/49.0D00)**49.0D00)

C  CWK: prefactor de la correccion Wigner-Kirkwood = lambda_B^2/(24*pi)
      CWK = XLB2/24.0D00/PI

C  UWT0: valor del potencial total (HSMIE+WK) en el minimo XMIN.
C  Se usa como desplazamiento WCA para que u(XMIN)=0 exactamente.
      UX0  = CTE*(XMIN**(-50.0D0)-XMIN**(-49.0D0))
      UWK0 = 50.0D0*49.0D0*XMIN**(-52.0D0)
     +     - 49.0D0*48.0D0*XMIN**(-51.0D0)
      UWT0 = CTE*CWK*UWK0 + UX0

C  Reescala el desplazamiento máximo a unidades de la caja
      DISPL=DISPL*S

C  Escritura de dimensiones convertidas
      WRITE(6,108) XC,YC,ZC
      WRITE(7,108) XC,YC,ZC
      WRITE(6,109) XL,YL,ZL
      WRITE(7,109) XL,YL,ZL
      WRITE(6,110) S
      WRITE(7,110) S

C  --- Generación de la configuración inicial ---
C
C  RX(I), RY(I), RZ(I): coordenadas del centro de la particula I
C  en unidades de la caja (valores entre 0 y 1 en x).
C
C  Si NRUN=0: coloca las partículas en una red cubica replicando
C  la celda unitaria NAC*NCX*NCY*NCZ veces.
      IF (NRUN.EQ.0.0D00) THEN
      I=1
      DO 1 IZ=1,NCZ
      DO 91 J=1,NAC
      DO 81 IX=1,NCX
      DO 71 IY=1,NCY
      RX(I)=((IX-1.0D00)+XA(J))/XNCX
      RY(I)=((IY-1.0D00)+YA(J))*YC/XNCY
      RZ(I)=((IZ-1.0D00)+ZA(J))*ZC/XNCZ
      I=I+1.0D00
   71 CONTINUE
   81 CONTINUE
   91 CONTINUE
    1 CONTINUE
      END IF

C  Si NRUN=1: lee posiciones previas del archivo mc.old
C  Si NRUN=0: salta la lectura (ya se generó la red arriba)
      IF (NRUN.EQ.0.0D00) GOTO 2
      OPEN(UNIT=3,FILE='mc.old',STATUS='OLD')
      DO 11 I=1,N
      READ(3,*) RX(I),RY(I),RZ(I)
   11 CONTINUE

C  Imprime las primeras 20 posiciones para verificación visual
    2 DO 44 I=1,20
      WRITE(6,115) RX(I),RY(I),RZ(I)
      WRITE(7,115) RX(I),RY(I),RZ(I)
   44 CONTINUE

C  --- Formatos de salida ---
  100 FORMAT(1X,'*********************************************',/
     +   ' ***SIMULACION MONTE CARLO NVT QHS (MIE(50,49)-WK)***',/
     +         ,' *********************************************',/,/)
  101 FORMAT(1X,' N= ',I6,' RHO= ',F10.8,' TEMP= ',F10.5,
     + ' XLB=',F10.5,' RC=',F10.8,/,/)
 1011 FORMAT(1X,' NRUN=',I4,' NMOVE=',I10,' NSUB=',I9,/,/)
 1012 FORMAT(1X,' NAC,NCX,NCY,NCZ=',4I4,/,/)
 1013 FORMAT(1X,' DISPL=',F10.8,/,/)
  102 FORMAT(1X,'INICIO DE UNA RED',/)
  103 FORMAT(1X,'INICIO DE CONFIGURACION PREVIA',/)
  108 FORMAT(1X,'TAMANO DE CAJA EN UNIDADES DE PROGRAMA :',/
     +         ,8X,'XC=',F9.5,' YC=',F9.5,' ZC=',F9.5,/)
  109 FORMAT(1X,'TAMANO DE CAJA EN UNIDADES DE SIGMA:  ',/
     +         ,8X,'XL=',F9.5,' YL=',F9.5,' ZL=',F9.5,/)
  110 FORMAT(1X,'SIGMA EN UNIDADES DE LA CAJA:',4X,F10.8,/)
  111 FORMAT(1X,'NUMERO PREVIO DE CONFIGURACIONES',F12.0,/)
  112 FORMAT(1X,'CONFIGURACION INICIAL')
  115 FORMAT(1X,3(1X,F10.8))
      RETURN
      END

C ===========================================================================
C  SUBROUTINE MONTECARLO
C
C  Proposito:
C    Ejecutar el ciclo de Metropolis NVT.
C    En cada paso intenta mover una partícula aleatoria y decide si
C    aceptar el movimiento según el criterio de Metropolis.
C
C  Algoritmo (por paso):
C    1. Elegir particula I al azar
C    2. Proponer nueva posición dentro de un cubo de lado DISPL
C    3. Aplicar condiciones periódicas de frontera
C    4. Calcular DENERG = U(posición nueva) - U(posición vieja)
C    5. Si DENERG <= 0: aceptar siempre
C       Si DENERG >  0: aceptar con probabilidad exp(-DENERG/TEMP)
C    6. Si se acepta: actualizar posición y energía total
C    7. Acumular estadísticas; cada NSUB pasos escribir subpromedio
C
C  Observables acumulados:
C    UAV   = energia interna promedio por particula
C    CVAV  = calor especifico por particula = (<U^2> - <U>^2*N^2)/(TEMP^2*N)
C    std1  = desviación estándar de UAV entre subpromedios
C    std2  = desviación estándar de CVAV entre subpromedios
C ===========================================================================
      SUBROUTINE MONTECARLO
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"
      open(unit=8,file='uav.dat')

C  Semilla inicial para el generador de numeros aleatorios RAN2.
C  Debe ser un entero negativo para inicializar la secuencia.
      ISEED=-123456789

C  --- Cálculo de la energía inicial ---
C  UPHS: energía total clásica HS continuo (potencial Mie)
C  UPWK: energía total corrección cuántica WK
      CALL ENERGIA(UPHS,UPWK)
      UPERPHS=UPHS/XN
      UPERPWK=UPWK/XN
      UPERP=UPERPHS+UPERPWK
      UTOTHS=UPHS
      UTOTWK=UPWK
      UTOT=UTOTHS+UTOTWK

C  --- Inicialización de contadores ---
      USUBAVHS=0.0D00
      USUBAVWK=0.0D00
      NAVE = 0
      NCOUNT=0
      NACCPT=0
      NSUB0=NSUB
      NFDR0 = NFDR

      WRITE(6,101)UPERP
      WRITE(7,101)UPERP

C  --- Energia teorica de referencia (ec. de estado Serna-Gil-Villegas 2016) ---
C  Se calcula para comparación, no se usa en el ciclo MC.
C  XLB2 viene del COMMON (calculado en COMIENZO).
C  eta = fracción de empaquetamiento = rho*pi/6
      et = rho*pi/6.d0
      et2 = et*et
      de1 = 1.6593854484d0
      de2 = -1.0927115150d0
      de3 = -1.1188233921d0
C  etq: fracción de empaquetamiento cuántico efectivo
      etq = (1.d0+de1*XLB)*et + (de2*XLB + de3*XLB2)*et2
      etq2 = etq*etq
      etq3 = etq2*etq
      detq = 1.d0-etq
      detq2 = detq*detq
      detq3 = detq2*detq
C  zqhs: factor de compresibilidad de la ecuación de estado cuántica HS
      zqhs = 1.d0+etq+etq2-etq3
      zqhs = zqhs/detq3 -1.d0
      dxletq = de1*et + (de2 + 2.d0*de3*XLB)*et2
C  UQHST: energia teorica por particula (en unidades de kT)
      UQHST = (zqhs/etq)*dxletq*dsqrt(TEMP)
      UQHST = 0.5d0*UQHST/dsqrt(2.d0*pi)

C ===========================================================================
C  CICLO DE METROPOLIS
C  Etiqueta 1: inicio de un paso MC
C  Etiqueta 6: fin de la corrida (NCOUNT >= NMOVE)
C ===========================================================================

C  --- Selección de partícula al azar ---
    1 I=INT(RAN2(ISEED)*N)+1
      NCOUNT=NCOUNT+1

C  Guarda la posición actual de la partícula I
      RXI = RX(I)
      RYI = RY(I)
      RZI = RZ(I)

C  --- Propuesta de movimiento ---
C  Desplazamiento uniforme dentro de [-DISPL/2, +DISPL/2] en cada eje
      XNEW=RXI+DISPL*(RAN2(ISEED)-0.5D00)
      YNEW=RYI+DISPL*(RAN2(ISEED)-0.5D00)
      ZNEW=RZI+DISPL*(RAN2(ISEED)-0.5D00)

C  Condiciones periódicas de frontera: si la partícula sale por un
C  lado de la caja, entra por el lado opuesto.
      IF(XNEW.GT.1.0D00) THEN
      XNEW=XNEW-1.0D00
      ELSE IF (XNEW.LT.0.0D00) THEN
      XNEW=XNEW+1.0D00
      END IF
      IF (YNEW.GT.YC) THEN
      YNEW=YNEW-YC
      ELSE IF (YNEW.LT.0.0D00) THEN
      YNEW=YNEW+YC
      END IF
      IF (ZNEW.GT.ZC) THEN
      ZNEW=ZNEW-ZC
      ELSE IF (ZNEW.LT.0.0D00) THEN
      ZNEW=ZNEW+ZC
      END IF

C  --- Calculo de energía en la posición nueva ---
C  UHSNEW: energia de la particula I con sus N-1 vecinas (parte HS)
C  UWKNEW: idem para la corrección WK
      UHSNEW=0.0D00
      UWKNEW=0.0D00

      DO 2 J=1,N
      IF (J.EQ.I) GOTO 2

C  Vector separación partícula J - posición nueva de I
      X=RX(J)-XNEW
      Y=RY(J)-YNEW
      Z=RZ(J)-ZNEW

C  Convención de imagen mínima: usar la replica mas cercana de J
      IF (X.GT.0.5D00) THEN
      X=X-1.0D00
      ELSE IF (X.LT.-0.5D00) THEN
      X=X+1.0D00
      END IF
      IF (Y.GT.Y2) THEN
      Y=Y-YC
      ELSE IF (Y.LT.-Y2) THEN
      Y=Y+YC
      END IF
      IF (Z.GT.Z2) THEN
      Z=Z-ZC
      ELSE IF (Z.LT.-Z2) THEN
      Z=Z+ZC
      END IF

      RR=X*X+Y*Y+Z*Z

C  Solo evalúa el potencial si el par esta dentro del radio de corte XMIN.
C  Las potencias se calculan por multiplicaciones sucesivas para eficiencia:
C    SR2  = (sigma/r)^2
C    SR10 = SR2^5
C    SR20 = SR10^2
C    SR50 = SR20^2 * SR10   = (sigma/r)^50
C    SR49 = SR50 / SR        = (sigma/r)^49
C    SR51 = SR50 * SR        = (sigma/r)^51
C    SR52 = SR50 * SR2       = (sigma/r)^52
      IF (RR.LT.XMIN2) THEN
        SR2=SS/RR
        SR=dsqrt(SR2)
        SR10=SR2*SR2*SR2*SR2*SR2
        SR20=SR10*SR10
        SR50=SR20*SR20*SR10
        SR49=SR20*SR20*SR2*SR2*SR2*SR2*SR
        SR51=SR50*SR
        SR52=SR50*SR2
C  Contribución clásica HS: u_HS = CTE*(sr50 - sr49) + 1
C  El +1 desplaza el potencial para que sea cero en el corte
        UHS = CTE*(SR50-SR49) + 1.d0
C  Corrección WK: u_WK = CTE*CWK*(2450*sr52 - 2352*sr51) - UWT0 - 1
C  Los coeficientes 2450=50*49 y 2352=49*48 vienen de nabla^2 u(r)
        UWK = CTE*CWK*(2450.0D00*SR52-2352.0D00*SR51) - UWT0-1.d0
      ELSE
       UHS = 0.d0
       UWK = 0.d0
      ENDIF

      UHSNEW=UHSNEW+UHS
      UWKNEW=UWKNEW+UWK
    2 CONTINUE

C  --- Calculo de energía en la posición vieja ---
      UHSOLD=0.0D00
      UWKOLD=0.0D00

      DO 4 J=1,N
      IF (J.EQ.I) GOTO 4

      X=RX(J)-RXI
      Y=RY(J)-RYI
      Z=RZ(J)-RZI

      IF (X.GT.0.5D00) THEN
      X=X-1.0D00
      ELSE IF (X.LT.-0.5D00) THEN
      X=X+1.0D00
      END IF
      IF (Y.GT.Y2) THEN
      Y=Y-YC
      ELSE IF (Y.LT.-Y2) THEN
      Y=Y+YC
      END IF
      IF (Z.GT.Z2) THEN
      Z=Z-ZC
      ELSE IF (Z.LT.-Z2) THEN
      Z=Z+ZC
      END IF

      RR=X*X+Y*Y+Z*Z

      IF (RR.LT.XMIN2) THEN
        SR2=SS/RR
        SR=dsqrt(SR2)
        SR10=SR2*SR2*SR2*SR2*SR2
        SR20=SR10*SR10
        SR50=SR20*SR20*SR10
        SR49=SR20*SR20*SR2*SR2*SR2*SR2*SR
        SR51=SR50*SR
        SR52=SR50*SR2
        UHS = CTE*(SR50-SR49) +1.d0
        UWK = CTE*CWK*(2450.0D00*SR52-2352.0D00*SR51)-UWT0 -1.d0
      ELSE
       UHS = 0.d0
       UWK = 0.d0
      ENDIF

      UHSOLD=UHSOLD+UHS
      UWKOLD=UWKOLD+UWK
    4 CONTINUE

C  --- Criterio de Metropolis ---
      DENERGHS = UHSNEW-UHSOLD
      DENERGWK = UWKNEW-UWKOLD
      DENERG = DENERGHS+DENERGWK

C  Si DENERG <= 0: el movimiento baja la energía, se acepta siempre
      IF (DENERG.LE.0.0D00) GOTO 5

C  Si DENERG > 0: se acepta con probabilidad exp(-DENERG/T*)
C  NOTA: se usa T*=1.5 fijo en lugar de la variable TEMP.
C  Temperatura de simulación del potencial
C  Fija por construcción: hace que Mie(50,49) imite esferas duras
C  Para usar la temperatura del input, reemplazar 1.5d0 por TEMP.
      RND = RAN2(ISEED)
      BOLTZ= DEXP(-DENERG/1.5d0)
      IF (RND.GT.BOLTZ) GOTO 3

C  Movimiento aceptado: actualizar posición y energía total
    5 RX(I)=XNEW
      RY(I)=YNEW
      RZ(I)=ZNEW
      UTOTHS=UTOTHS+DENERGHS
      UTOTWK=UTOTWK+DENERGWK
      UTOT=UTOTHS+UTOTWK
      NACCPT=NACCPT+1

C  --- Acumulación de estadísticas (independiente de aceptación) ---
    3 ACC(1)=ACC(1)+1.0D00       ! contador de pasos
      ACC(2)=ACC(2)+UTOTHS       ! suma de energia HSMIE
      ACC(12)=ACC(12)+UTOTWK     ! suma de corrección WK
      ACC(5)=ACC(5)+UTOT*UTOT    ! suma de U^2 (para calor especifico)
      USUBAVHS=USUBAVHS+UTOTHS
      USUBAVWK=USUBAVWK+UTOTWK
      USUBAV = USUBAVHS + USUBAVWK

      XNTEST=DFLOAT(NCOUNT)
      XNACEPT=DFLOAT(NACCPT)

C  Muestreo del histograma de g(r) cada NFDR pasos
      IF (NCOUNT.EQ.NFDR0.AND.LFDR) CALL GOFR

C  Fin de corrida
      IF (NCOUNT.GE.NMOVE) GOTO 6

C  Continuar sin escribir hasta el próximo subpromedio
      IF (NCOUNT.LT.NSUB) GOTO 1

C  --- Cálculo y escritura de subpromedios ---
      NAVE = NAVE + 1
      xnave = dfloat(NAVE)

C  Energia promedio por particula en este subpromedio
      UAVHS=USUBAVHS/(XN*NCOUNT)
      UAVWK=USUBAVWK/(XN*NCOUNT)
      UAV=UAVHS+UAVWK

C  Calor especifico por particula: CV/N = (<U^2> - N^2*<U>^2) / (T^2 * N)
      UAV2 = ACC(5)/XNTEST
      CVAV = (UAV2 - XN*XN*UAV*UAV)/TEMP/TEMP
      CVAV = CVAV/XN

C  Desviación estándar de UAV y CVAV entre subpromedios
      ACC(10) = ACC(10) + UAV
      ACC(18) = ACC(18) + UAV*UAV
      ACC(14) = ACC(14) + CVAV
      ACC(16) = ACC(16) + CVAV*CVAV
      xprom1 = ACC(10)/xnave
      xprom2 = ACC(18)/xnave
      xprom3 = ACC(14)/xnave
      xprom4 = ACC(16)/xnave
      std1 = xprom2 - xprom1*xprom1
      std2 = xprom4 - xprom3*xprom3
C  max(0,x) protege contra valores ligeramente negativos por redondeo
C  que pueden ocurrir cuando NAVE=1 (primer subpromedio)
      std1 = dsqrt(max(0.0d0, std1))
      std2 = dsqrt(max(0.0d0, std2))

      WRITE(6,102) NCOUNT,XNACEPT/XNTEST,UAVHS,UAVWK,UAV,std1,UQHST,
     &CVAV,std2
      WRITE(7,102) NCOUNT,XNACEPT/XNTEST,UAVHS,UAVWK,UAV,std1,UQHST,
     &CVAV,std2
      WRITE(8,200) NCOUNT,UAVHS,UAVWK,UAV,std1,CVAV,std2

C  Actualiza el umbral del próximo subpromedio
      NSUB=NSUB+NSUB0
      GOTO 1

C  --- Fin de la corrida (NCOUNT >= NMOVE) ---
       NAVE = NAVE + 1
       xnave = dfloat(NAVE)
    6  UAVHS=USUBAVHS/(XN*NCOUNT)
       UAVWK=USUBAVWK/(XN*NCOUNT)
       UAV = UAVHS + UAVWK
       UAV2=ACC(5)/XNTEST
       UAVRUN=(ACC(2)+ACC(12))/XN/ACC(1)  ! promedio de toda la corrida
       CVAV = (UAV2 - XN*XN*UAV*UAV)/TEMP/TEMP
       CVAV = CVAV/XN

      xnave = dfloat(NAVE)
      ACC(10) = ACC(10) + UAV
      ACC(18) = ACC(18) + UAV*UAV
      ACC(14) = ACC(14) + CVAV
      ACC(16) = ACC(16) + CVAV*CVAV
      xprom1 = ACC(10)/xnave
      xprom2 = ACC(18)/xnave
      xprom3 = ACC(14)/xnave
      xprom4 = ACC(16)/xnave
      std1 = xprom2 - xprom1*xprom1
      std2 = xprom4 - xprom3*xprom3
C  max(0,x) protege contra valores ligeramente negativos por redondeo
      std1 = dsqrt(max(0.0d0, std1))
      std2 = dsqrt(max(0.0d0, std2))

      WRITE(6,102) NCOUNT,XNACEPT/XNTEST,UAVHS,UAVWK,UAV,std1,UQHST,
     &CVAV,std2
      WRITE(7,102) NCOUNT,XNACEPT/XNTEST,UAVHS,UAVWK,UAV,std1,UQHST,
     &CVAV,std2
      WRITE(8,200) NCOUNT,UAVHS,UAVWK,UAV,std1,CVAV,std2

      WRITE(6,103) ACC(1)
      WRITE(7,103) ACC(1)
      WRITE(6,104) UAVRUN
      WRITE(7,104) UAVRUN
      close(8)
      RETURN

C  Mensaje de error: partículas traslapadas (solo si hay solapamiento)
   42 WRITE(6,105) I,J,RR
      WRITE(6,106) SQRT(RR),S
      WRITE(6,*) RX(I),RY(I),RZ(I)
      WRITE(6,*) RX(J),RY(J),RZ(J)

C  --- Formatos de salida ---
  101 FORMAT(/,1X,'inicio de corrida',/,1X,'energia inicial',G16.8,/)
  102 FORMAT(1X,'NMOV  =',I12,' AC  =',F8.4,' UHS =',F8.4,
     &' UWK =',F8.4,' UAV =',F8.4, ' std1=',F8.4, ' UTEO=',F8.4,
     &' CV =',F8.4,
     &' std2=',F8.4,/)
  200 FORMAT(I12, F10.4, F10.4, F10.4, F10.4, F10.4, F10.4)
  103 FORMAT(1X,'Num total de configuraciones desde inicio',F12.0,/)
  104 FORMAT(1X,'Energia promedio desde el inicio',F10.4,/)
  105 FORMAT(1X,'*****particulas traslapadas*****',' I=',I4,' J=',I4,
     + ' R**2=',G16.8,/)
  106 FORMAT(1X,'R=',G16.8,' S=',G16.8,/)
      RETURN
      END

C ===========================================================================
C  SUBROUTINE ENERGIA(UTOTHS, UTOTWK)
C
C  Proposito:
C    Calcular la energia potencial total del sistema.
C    Bucle doble O(N^2) sobre todos los pares (i,j) con j > i.
C
C  Argumentos de salida:
C    UTOTHS -> energía total clásica (contribución Mie/HS)
C    UTOTWK -> energía total corrección cuántica (Wigner-Kirkwood)
C
C  Nota: esta subrutina se llama una vez al inicio para obtener la
C  energía inicial. Durante el ciclo MC se actualiza la energía
C  de forma incremental (solo la particula movida).
C ===========================================================================
      SUBROUTINE ENERGIA(UTOTHS,UTOTWK)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"

      UTOTHS=0.0D0
      UTOTWK=0.0D0

C  Todas las constantes del potencial (XLB2, XMIN, XMIN2, CWK, CTE, UWT0)
C  vienen del COMMON: fueron calculadas una vez en COMIENZO.

C  Bucle doble sobre todos los pares únicos (i < j)
      DO 4 I=1,N-1
      RXI = RX(I)
      RYI = RY(I)
      RZI = RZ(I)
      DO 41 J=I+1,N

C  Vector separación y convención de imagen mínima
      X=RX(J)-RXI
      Y=RY(J)-RYI
      Z=RZ(J)-RZI

      IF (X.GT.0.5D00) THEN
      X=X-1.0D00
      ELSE IF (X.LT.-0.5D00) THEN
      X=X+1.0D00
      END IF
      IF (Y.GT.Y2) THEN
      Y=Y-YC
      ELSE IF (Y.LT.-Y2) THEN
      Y=Y+YC
      END IF
      IF (Z.GT.Z2) THEN
      Z=Z-ZC
      ELSE IF (Z.LT.-Z2) THEN
      Z=Z+ZC
      END IF

      RR=X*X+Y*Y+Z*Z

C  Evaluación del potencial (solo si el par esta dentro del corte)
      IF(RR.LT.XMIN2) THEN
        SR2=SS/RR
        SR=dsqrt(SR2)
        SR10=SR2*SR2*SR2*SR2*SR2
        SR20=SR10*SR10
        SR50=SR20*SR20*SR10
        SR49=SR20*SR20*SR2*SR2*SR2*SR2*SR
        SR51=SR50*SR
        SR52=SR50*SR2
        UHS = CTE*(SR50-SR49) + 1.d0
        UWK = CTE*CWK*(2450.0D00*SR52-2352.0D00*SR51) - UWT0-1.d0
      ELSE
        UHS = 0.d0
        UWK = 0.d0
      ENDIF

      UTOTHS = UTOTHS +UHS
      UTOTWK = UTOTWK +UWK
   41 CONTINUE
    4 CONTINUE

      RETURN
      END

C ===========================================================================
C  SUBROUTINE GOFR
C
C  Proposito:
C    Actualizar el histograma de pares G(L) para la función de
C    distribución radial g(r). Se llama cada NFDR pasos MC.
C
C  Metodo:
C    Para cada par (i,j) con j > i, calcula la distancia r y la
C    asigna al bin L del histograma según:
C      L = INT((r - 0.75*sigma) * XHIST / (1 - 0.75*sigma)) + 1
C    El origen del histograma esta en r = 0.75*sigma (no en sigma)
C    para capturar la region de contacto del potencial.
C
C  La normalización para obtener g(r) se realiza en SUBROUTINE RADIAL.
C ===========================================================================
      SUBROUTINE GOFR
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"

C  Incrementa el contador de muestras (usado en la normalización)
      ACC(3)=ACC(3)+1.0D00

C  Radio máximo de inclusión en el histograma: mitad de la caja
      RMAX=0.5D00*0.5D00

      DO 1 I=1,N-1
      DO 11 J=I+1,N

      X=RX(I)-RX(J)
      Y=RY(I)-RY(J)
      Z=RZ(I)-RZ(J)

C  Convención de imagen mínima
      IF (X.GT.0.5D00) THEN
      X=X-1.0D00
      ELSE IF (X.LT.-0.5D00) THEN
      X=X+1.0D00
      END IF
      IF (Y.GT.Y2) THEN
      Y=Y-YC
      ELSE IF (Y.LT.-Y2) THEN
      Y=Y+YC
      END IF
      IF (Z.GT.Z2) THEN
      Z=Z-ZC
      ELSE IF (Z.LT.-Z2) THEN
      Z=Z+ZC
      END IF

      RR=X*X+Y*Y+Z*Z

C  Descarta pares fuera del radio máximo (r > L/2)
      IF (RR.GT.RMAX) GOTO 1
      R=DSQRT(RR)

C  Asignación al bin del histograma
C  Se verifica que L este dentro del rango válido [1, NG]
C  para evitar acceso fuera del arreglo G si r < 0.75*sigma
      L=INT((R-0.75d0*S)*XHIST/(1.d0-0.75d0*S))+1
      IF (L.LT.1 .OR. L.GT.NG) GOTO 11
      G(L)=G(L)+1.0D00

   11 CONTINUE
    1 CONTINUE

C  Actualiza el próximo paso de muestreo
      NFDR0=NFDR+NFDR0

      RETURN
      END

C ===========================================================================
C  SUBROUTINE RADIAL
C
C  Proposito:
C    Normalizar el histograma G(L) para obtener g(r) y calcular
C    el factor de compresibilidad Z via la ecuación de virial.
C
C  Normalización de g(r):
C    GR(L) = G(L) / (CONST * (Rc^3 - Rb^3))
C    CONST = (2/3) * N^2 * pi * Nsamples / V
C    El denominador es el numero de pares esperado en el cascaron
C    esferico [Rb, Rc] para un gas ideal a la misma densidad.
C
C  Factor de compresibilidad (ecuación de virial):
C    Z = 1 + (2*pi/3) * rho * integral( g(r) * r * du/dr * r^2 dr )
C    Se separan las contribuciones repulsiva (scol) y atractiva (xlcol).
C    Los parámetros de colisión <s> y <l> son cocientes de integrales.
C ===========================================================================
      SUBROUTINE RADIAL
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"
      OPEN(UNIT=9,FILE='rdf.dat',STATUS='NEW')

      NHISTG=nint(XHIST)
      XNAV=ACC(3)                  ! numero de muestras de g(r)
      IF(XNAV.EQ.0.0D00)RETURN     ! no hay muestras, salir

C  Anchura del bin en unidades de sigma
      DELR=(1.d0-0.75*S)*XL/XHIST

C  Volumen de la caja en unidades de sigma^3
      VOLME=XL*YL*ZL

C  Constante de normalización
      CONST=2.0D00/3.0D00*XN*XN*PI*XNAV/VOLME

C  Numero de bins hasta r = L/2 (radio de corte de la caja)
      NEND=INT((0.5D00*XC-0.75d0*S)*XHIST/(1.d0-0.75d0*S))

C  Inicialización de acumuladores para Z
      scol = 0.d0
      fr = 0.d0
      xlcol = 0.d0
      fa = 0.d0

C  XLB2, CWK, CTE vienen del COMMON (calculados en COMIENZO).

      DO 1 L=1,NEND
      X=FLOAT(L)

C  Radio central del bin L (en unidades de sigma)
      R=(1.0D00-0.75d0*S)*(2.0D00*X-1.0D00)*XL/(XHIST*2.0D00)
C  Radio del borde interior del bin
      RB=(1.0D00-0.75d0*S)*(X-1.0D00)*XL/XHIST
C  Desplazamiento al origen real del histograma (r = 0.75*sigma)
      R=R+0.75D00
      RB=RB+0.75D00

C  Potencias de R para evaluar la derivada del potencial
      RSQ=R*R
      RS3=R*R*R
      RS5=RSQ*RS3
      RS10=RS5*RS5
      RS20=RS10*RS10
      RS40=RS20*RS20
      RS50=RS40*RS10
      RS51=RS50*R
      RS52=RS50*RSQ
      RS53=RS50*RS3

      RBSQ=RB*RB
      RBCU=RBSQ*RB
      RC=RB+DELR
      RC3=RC*RC*RC

C  Normalización del bin: divide por el número de pares ideal en [Rb,Rc]
      GR=G(L)/(CONST*(RC3-RBCU))
      write(9,*)R,GR

C  Derivada del potencial respecto a r: du/dr = CTE * (DU1 + DU2)
C    DU1: termino clásico HS
C    DU2: termino de corrección WK
      DU1 = 49.0D00/RS50 - 50.0D00/RS51
      DU2 = 392.0D00*CWK*(306.0D00/RS52 - 325.0D00/RS53)
      DU = CTE*(DU1 + DU2)

C  Contribuciones a la integral de virial
C  Si DU < 0 (fuerza atractiva): contribuye a la rama atractiva
C  Si DU > 0 (fuerza repulsiva): contribuye a la rama repulsiva
      SCOL3 = GR*DU*RS3*DELR
      SCOL2 = GR*DU*RSQ*DELR

      if (DU.lt.0.d0) then
        scol  = scol  + SCOL3
        fr    = fr    + SCOL2
	  else
        xlcol = xlcol + SCOL3
        fa    = fa    + SCOL2
      endif

1     CONTINUE

C  Parametros de colisión: <s> repulsivo, <l> atractivo
      sfin     = scol/fr
C  Para un potencial puramente repulsivo (WCA), la rama atractiva
C  puede no tener contribuciones. Se evita la división por cero.
      if (fa.gt.0.d0) then
        xlcolfin = xlcol/fa
        freqa    = 24.d0*fa*RHO
      else
        xlcolfin = 0.d0
        freqa    = 0.d0
      endif

C  Frecuencia de colisión repulsiva
      freqr = -24.d0*fr*RHO

C  Factor de compresibilidad Z = PV/(NkT)
      zeta = 2.d0*pi*(sfin*freqr - xlcolfin*freqa)/3.d0
      zeta = zeta + 1.d0

      write(6,20)freqr,freqa,sfin,xlcolfin,zeta
   20 FORMAT(1X,'NUR  =',F8.5,'  NUA  =',F8.5,'  <s> =',F8.5,
     & '  <l>=',F8.5, '  Z=',f10.5/)

      RETURN
      END

C ===========================================================================
C  SUBROUTINE FINAL
C
C  Proposito:
C    Guardar la configuración final del sistema en mc.new para permitir
C    reanudar la simulación en una corrida posterior (usar como mc.old
C    con NRUN=1).
C ===========================================================================
      SUBROUTINE FINAL
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      include "mc.inc"

      OPEN(UNIT=4,FILE='mc.new',STATUS='NEW')

C  Escribe las coordenadas finales de todas las partículas
      DO 55 I=1,N
      WRITE(4,*) RX(I),RY(I),RZ(I)
   55 CONTINUE

C  Imprime la configuración final en pantalla y en mc.dat
      WRITE(6,101)
      WRITE(7,101)
      DO 88 I=1,N
      WRITE(6,*) RX(I),RY(I),RZ(I)
      WRITE(7,*) RX(I),RY(I),RZ(I)
   88 CONTINUE

 101  FORMAT(/,/,2X,'configuracion final',/,10X,'RX',20X,'RY',20X,
     + 'RZ',/)

      close(7)
      close(4)
      RETURN
      END

C ===========================================================================
C  FUNCTION RAN2(IDUM)
C
C  Proposito:
C    Generador de números aleatorios uniformes en (0,1) de alta calidad.
C    Implementación de RAN2 de Numerical Recipes (Press et al., 2nd Ed.).
C
C  Algoritmo:
C    Combinación de dos generadores congruentes lineales (periodo > 2e18)
C    con una tabla de barajeado de 32 entradas para eliminar correlaciones
C    de largo alcance.
C
C  Uso:
C    Inicializar con IDUM < 0 (por ejemplo IDUM=-123456789).
C    Llamadas sucesivas con el mismo IDUM devuelven la misma secuencia.
C
C  Parametros:
C    IM1, IM2  : módulos de los dos generadores (primos grandes)
C    IA1, IA2  : multiplicadores
C    IQ1, IQ2  : cocientes para el método de Schrage (evita overflow)
C    IR1, IR2  : residuos para el metodo de Schrage
C    NTAB      : tamaño de la tabla de barajeado
C    EPS       : evita que la salida sea exactamente 1.0
C ===========================================================================
        FUNCTION RAN2(IDUM)
        IMPLICIT DOUBLE PRECISION(A-H,O-Z)
        PARAMETER (IM1=2147483563,IM2=2147483399,AM=1.0D00/IM1,
     &  IMM1=IM1-1,IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     &  IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB)
        PARAMETER(EPS=1.2D-14,RNMX=1.0D00-EPS)
        DIMENSION IV(NTAB)
        SAVE IV,IY,IDUM2
        DATA IDUM2/123456789/,IV/NTAB*0/,IY/0/

C  Inicialización: se ejecuta solo la primera vez (IDUM <= 0)
        IF (IDUM.LE.0)THEN
          IDUM=MAX(-IDUM,1)
          IDUM2=IDUM
          DO J=NTAB+8,1,-1
            K=IDUM/IQ1
            IDUM=IA1*(IDUM-K*IQ1)-K*IR1
            IF(IDUM.LT.0)IDUM=IDUM+IM1
            IF(J.LE.NTAB)IV(J)=IDUM
          ENDDO
          IY=IV(1)
        ENDIF

C  Avanza el primer generador congruente lineal (metodo de Schrage)
        K=IDUM/IQ1
        IDUM=IA1*(IDUM-K*IQ1)-K*IR1
        IF(IDUM.LT.0)IDUM=IDUM+IM1

C  Avanza el segundo generador
        K=IDUM2/IQ2
        IDUM2=IA2*(IDUM2-K*IQ2)-K*IR2
        IF(IDUM2.LT.0)IDUM2=IDUM2+IM2

C  Extrae un valor de la tabla y lo reemplaza (barajeado de Bays-Durham)
        J=1+IY/NDIV
        IY=IV(J)-IDUM2
        IV(J)=IDUM
        IF(IY.LT.1)IY=IY+IMM1

C  Convierte a real en (0, 1-EPS)
        RAN2=MIN(AM*IY,RNMX)
        RETURN
        END
