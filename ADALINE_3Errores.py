import numpy as np

# Función de ADALINE
def adaline(entrenamiento, salidas, tasa_aprendizaje, epocas, tipo_error="simple", x_factor=0.25, pesos_iniciales=None, sesgo_inicial=None):
    entrenamiento = np.array(entrenamiento)
    salidas = np.array(salidas)
    num_entradas = entrenamiento.shape[1]

    # Inicializar pesos aleatorios si no se proporcionan
    if pesos_iniciales is None:
        pesos = np.random.rand(num_entradas)
    else:
        pesos = np.array(pesos_iniciales, dtype=float)

    # Sesgo aleatorio si no se proporciona
    if sesgo_inicial is None:
        sesgo = np.random.rand()
    else:
        sesgo = sesgo_inicial

    print("Pesos iniciales:", np.round(pesos,4), "Sesgo inicial:", round(sesgo,4))

    # Entrenamiento
    for epoca in range(epocas):
        print("\nÉpoca", epoca + 1)
        errores_epoca = []
        for i in range(len(entrenamiento)):
            x = entrenamiento[i]
            d = salidas[i]

            # Salida lineal
            z = np.dot(x, pesos) + sesgo
            y = z

            # Calcular errores
            error_simple = d - y
            error_cuadrado = (d - y) ** 2
            error_escalado = x_factor * (d - y) ** 2

            # Elegir actualización según tipo de error
            if tipo_error == "simple":
                ajuste = error_simple
            elif tipo_error == "cuadrado":
                ajuste = error_cuadrado
            elif tipo_error == "escalado":
                ajuste = error_escalado
            else:
                ajuste = error_simple

            # Actualizar pesos y sesgo
            pesos = pesos + tasa_aprendizaje * ajuste * x
            sesgo = sesgo + tasa_aprendizaje * ajuste

            errores_epoca.append((error_simple, error_cuadrado, error_escalado))

            print("Entrada:", x, "D:", d, "Y:", round(y,4),
                  "Error:", round(error_simple,4),
                  "Error^2:", round(error_cuadrado,4),
                  "x*(Error^2):", round(error_escalado,4),
                  "Pesos:", np.round(pesos,4),
                  "Sesgo:", round(sesgo,4))
    return pesos, sesgo

# Función de predicción (adaptada a la consigna)
def predict(patron, pesos, sesgo):
    z = np.dot(patron, pesos) + sesgo
    return np.where(z >= 0, 1, -1)

# ------------------ PROGRAMA PRINCIPAL ------------------
print("=== ADALINE Completo ===")

# Entradas del usuario
num_entradas = int(input("Número de entradas por patrón: "))
num_patrones = int(input("Número de patrones: "))

entrenamiento = []
salidas = []

for i in range(num_patrones):
    fila = []
    for j in range(num_entradas):
        valor = float(input(f"Entrada {j+1} del patrón {i+1}: "))
        fila.append(valor)
    entrenamiento.append(fila)
    salida = float(input(f"Salida esperada del patrón {i+1}: "))
    salidas.append(salida)

tasa_aprendizaje = float(input("Tasa de aprendizaje: "))
epocas = int(input("Número de épocas: "))

# Tipo de error
print("Tipo de error:")
print("1. simple")
print("2. cuadrado")
print("3. escalado")
opcion_error = input("Seleccione tipo de error (simple/cuadrado/escalado): ").lower()
x_factor = 0.25
if opcion_error == "escalado":
    x_factor = float(input("Factor x para error escalado (ej: 0.25, 0.125, 0.5): "))

# Entrenar ADALINE
pesos_finales, sesgo_final = adaline(entrenamiento, salidas, tasa_aprendizaje, epocas,
                                     tipo_error=opcion_error, x_factor=x_factor)

print("\nPesos finales:", np.round(pesos_finales,4))
print("Sesgo final:", round(sesgo_final,4))

# Evaluación final con los mismos patrones de entrenamiento
print("\n=== Evaluación con predict ===")
for i, x in enumerate(entrenamiento):
    pred = predict(x, pesos_finales, sesgo_final)
    print("Entrada:", x, "Salida esperada:", salidas[i], "Predicción:", pred)

# Predicción libre
print("\n=== Predicción libre ===")
while True:
    try:
        nuevo = [float(input(f"Ingrese entrada {i+1} (o 's' para salir): ")) for i in range(num_entradas)]
        y_pred = predict(nuevo, pesos_finales, sesgo_final)
        print("Predicción:", y_pred)
    except:
        break
