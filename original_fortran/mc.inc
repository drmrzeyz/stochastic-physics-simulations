      PARAMETER (NPART=864,NACC=20,NG=1000)
      LOGICAL LFDR
      COMMON /BPOSITN/ RX(NPART),RY(NPART),RZ(NPART),ACC(NACC),G(NG)
C    XLB2,XLB3 : potencias de la longitud de onda termica
C    XMIN      : posicion del minimo del potencial efectivo (sigma)
C    XMIN2     : XMIN^2 * SS, radio de corte al cuadrado (unid. caja)
C    CTE       : prefactor Mie = 50*(50/49)^49
C    CWK       : prefactor WK  = lambda_B^2 / (24*pi)
C    UWT0      : valor del potencial en el minimo (desplazamiento WCA)
      COMMON /BCONSTR/ PI,ETA,RHO,XN,TEMP,DISPL,XLB,XHIST,
     +                 S,SS,XC,YC,ZC,
     +                 XL,YL,ZL,Y2,Z2,
     +                 XLB2,XLB3,XMIN,XMIN2,CWK,CTE,UWT0
      COMMON /BCONSTI/ N,NMOVE,NSUB,ISEED,NFDR,NFDR0,LFDR
