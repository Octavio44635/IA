import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# --- 1. DEFINICIÓN DE UNIVERSOS Y FPS ---
notas_examen = np.linspace(0, 100, 201)
conceptos = np.linspace(0, 10, 201)
notas_finales = np.linspace(0, 10, 201)

# Examen 
notas_examen_lo = fuzz.trapmf(notas_examen, [0, 0, 45, 55])
notas_examen_md = fuzz.trimf(notas_examen, [45, 65, 85])
notas_examen_hi = fuzz.trapmf(notas_examen, [75, 90, 100, 100])

# Concepto 
conceptos_lo = fuzz.trimf(conceptos, [0, 0, 5])
conceptos_md = fuzz.gaussmf(conceptos, 7, 1.5)
conceptos_hi = fuzz.trimf(conceptos, [8, 10, 10])

# Nota Final (7 clases)
NF_pert_muy_baja = fuzz.trimf(notas_finales, [0, 0, 3.5])
NF_pert_baja = fuzz.trimf(notas_finales, [2, 4.5, 5.5])
NF_pert_baja_media = fuzz.trimf(notas_finales, [4.5, 5.5, 6.5])
NF_pert_media = fuzz.trimf(notas_finales, [6.0, 7.5, 8.5])
NF_pert_media_alta = fuzz.trimf(notas_finales, [7.5, 8.5, 9.5])
NF_pert_alta = fuzz.trimf(notas_finales, [8.5, 9.5, 10])
NF_pert_sobresaliente = fuzz.trimf(notas_finales, [9.8, 10, 10])


reglas = [
    (notas_examen_lo, conceptos_lo, NF_pert_muy_baja),
    (notas_examen_lo, conceptos_md, NF_pert_baja),
    (notas_examen_lo, conceptos_hi, NF_pert_baja_media), 
 
    (notas_examen_md, conceptos_lo, NF_pert_baja_media),
    (notas_examen_md, conceptos_md, NF_pert_media),
    (notas_examen_md, conceptos_hi, NF_pert_media_alta),

    (notas_examen_hi, conceptos_lo, NF_pert_media),
    (notas_examen_hi, conceptos_md, NF_pert_alta),
    (notas_examen_hi, conceptos_hi, NF_pert_sobresaliente)
]

valores_prueba = [(10, 1.0), (10, 10.0),(45, 1.0), (45, 5.0),(50, 10.0),(55, 1.0), (55, 10.0),(65, 3.0), 
           (65, 7.0),(75, 10.0),(85, 1.0), (85, 5.0),(90, 10.0), (90, 3.0),(100, 1.0),(50, 7.0), 
           (80, 10.0),(60, 10.0), (70, 3.0),(98, 9.0)]

print("RESULTADOS CORREGIDOS")
print("{:<10} | {:<10} | {:<10} ".format("Examen", "Concepto", "Nota Final"))
print("-" * 33)

plt.ion() # Activa el modo interactivo para que los gráficos aparezcan de inmediato



# --- BUCLE DE SIMULACIÓN ---
for nota_examen, concepto in valores_prueba:
    
    # Fuzzificación
    NE_in_lo = fuzz.interp_membership(notas_examen, notas_examen_lo, nota_examen)
    NE_in_md = fuzz.interp_membership(notas_examen, notas_examen_md, nota_examen)
    NE_in_md = fuzz.interp_membership(notas_examen, notas_examen_md, nota_examen) 
    NE_in_hi = fuzz.interp_membership(notas_examen, notas_examen_hi, nota_examen)
    
    C_in_lo = fuzz.interp_membership(conceptos, conceptos_lo, concepto)
    C_in_md = fuzz.interp_membership(conceptos, conceptos_md, concepto)
    C_in_hi = fuzz.interp_membership(conceptos, conceptos_hi, concepto)

   
    # NE_in_lo, NE_in_md, NE_in_hi = NE_in_lo*2, NE_in_md2, NE_in_hi*2
    # C_in_lo, C_in_md, C_in_hi = C_in_lo*2, C_in_md2, C_in_hi*2
    
    activaciones = []
    
    
    for NE_pert, C_pert, NF_pert in reglas:
       
        if NE_pert is notas_examen_lo:
            NE_in = NE_in_lo
        elif NE_pert is notas_examen_md: 
            NE_in = NE_in_md
        else: 
            NE_in = NE_in_hi 
        
        if C_pert is conceptos_lo: 
            C_in = C_in_lo
        elif C_pert is conceptos_md: 
            C_in = C_in_md
        else: 
            C_in = C_in_hi 

        
        grado_activacion = np.fmin(NE_in, C_in)
        activacion_regla = np.fmin(grado_activacion, NF_pert)
        activaciones.append(activacion_regla)

    
    agregacion = np.fmax.reduce(activaciones)
    notas_final = fuzz.defuzz(notas_finales, agregacion, 'centroid')
    
    print(f"{nota_examen:<10} | {concepto:<10.2f} | {notas_final:.2f}")
    
    #comentar de aca si no se quieren los graficos para cada valor(puede explotar su pc)
    fig, ax2 = plt.subplots(figsize=(8, 3))
    
    # Graficando todas las FPs de salida para referencia
    ax2.plot(notas_finales, NF_pert_muy_baja, 'k--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_baja, 'b--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_baja_media, 'g--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_media, 'y--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_media_alta, 'm--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_alta, 'r--', linewidth=0.5)
    ax2.plot(notas_finales, NF_pert_sobresaliente, 'c--', linewidth=0.5)
    
    # Relleno del área agregada y línea de resultado
    ax2.fill_between(notas_finales, 0, agregacion, facecolor='Orange', alpha=0.7)
    nota_activation = fuzz.interp_membership(notas_finales, agregacion, notas_final)
    ax2.plot([notas_final, notas_final], [0, nota_activation], 'k', linewidth=1.5, alpha=0.9)
    ax2.set_title(f"Resultado E={nota_examen}, C={concepto:.2f} -> NF={notas_final:.2f}")
    plt.tight_layout()
    plt.show(block=False) 
    ##hasta aca




fig, ax_ex = plt.subplots(figsize=(8, 3))
ax_ex.plot(notas_examen, notas_examen_lo, 'b', linewidth=1.5, label='Baja (LO)')
ax_ex.plot(notas_examen, notas_examen_md, 'g', linewidth=1.5, label='Media (MD)')
ax_ex.plot(notas_examen, notas_examen_hi, 'r', linewidth=1.5, label='Alta (HI)')
ax_ex.set_title("1. Funciones de Pertenencia del Examen")
ax_ex.legend()
plt.tight_layout()
plt.show(block=False)


fig, ax_con = plt.subplots(figsize=(8, 3))
ax_con.plot(conceptos, conceptos_lo, 'b', linewidth=1.5, label='Regular (LO)')
ax_con.plot(conceptos, conceptos_md, 'g', linewidth=1.5, label='Bueno (MD)')
ax_con.plot(conceptos, conceptos_hi, 'r', linewidth=1.5, label='Excelente (HI)')
ax_con.set_title("2. Funciones de Pertenencia del Concepto")
ax_con.legend()
plt.tight_layout()
plt.show(block=False)


fig, ax_nf = plt.subplots(figsize=(8, 3))
ax_nf.plot(notas_finales, NF_pert_muy_baja, 'k', linewidth=1.5, label='M-Baja')
ax_nf.plot(notas_finales, NF_pert_baja, 'b', linewidth=1.5, label='Baja')
ax_nf.plot(notas_finales, NF_pert_baja_media, 'c', linewidth=1.5, label='Baja-Media')
ax_nf.plot(notas_finales, NF_pert_media, 'y', linewidth=1.5, label='Media')
ax_nf.plot(notas_finales, NF_pert_media_alta, "m", linewidth=1.5, label='Media-Alta')
ax_nf.plot(notas_finales, NF_pert_alta, 'r', linewidth=1.5, label='Alta')
ax_nf.plot(notas_finales, NF_pert_sobresaliente, 'g', linewidth=1.5, label='Sobresaliente.')
ax_nf.set_title("3. Funciones de Pertenencia de la Nota Final")
ax_nf.legend(ncol=3)
plt.tight_layout()
plt.show(block=False)


plt.ioff()
plt.show()