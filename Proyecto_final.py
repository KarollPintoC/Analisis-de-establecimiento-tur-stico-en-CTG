#%%
#Definicion de las librerias que usamos en el proyecto 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

#Leer el dataframe y estudios de sus datos
data_proyect=pd.read_csv('Registro_Nacional_de_Turismo_-_RNT_20250629.csv')
data_proyect.info()

# %%
#Convertir los datos que estaban de forma objeto pero que eran numeros a datos numericos 
columnas_a_convertir = ['NUM_EMP', 'HABITACIONES', 'CAMAS']
print("Iniciando limpieza y conversión...")
for col in columnas_a_convertir:
    print(f"Procesando columna: '{col}'")
    cleaned_col = data_proyect[col].astype(str).str.replace(',', '', regex=False)
    data_proyect[col] = pd.to_numeric(cleaned_col, errors='coerce').astype('Int64')
data_proyect.info()

# %%
#Hacer un filtrado para obtener sola la información del departamento de bolivar
df_bolivar=data_proyect[data_proyect["DEPARTAMENTO"]=="BOLIVAR"]
df_bolivar

# %%
# Vizualizar como se comporta los valores numericos del nuevo dataframe 
df_bolivar.describe()

# %%
#Visualizamos los codigos de municipios, departamentos y rnt, este caso eliminamos los codigos departamentos y rnt por que no son utiles para el estudio
cm=df_bolivar['COD_MUN'].unique()
print(f"Estos son los codigos de municipio {cm}, \nes decir que en total {len(cm)} de codigos de municipio")
cd=df_bolivar["COD_DPTO"].unique()
print(f"El codigo del departamento {cd}")
rnt=df_bolivar['CODIGO_RNT']
print(f"La cantidad de los codigos de rnt son {len(rnt)}")
#%%
df_bolivar.pop('COD_DPTO')
df_bolivar.pop('CODIGO_RNT')

# %%
# Agrupamos el codigo del municipio con el nombre 
df_bolivar.groupby("MUNICIPIO")["COD_MUN"].unique()
#%%
# Visualizamos cuantos municipios tenemos recolectados 
print(f"Hay {df_bolivar['COD_MUN'].nunique()} de 46 municipios que posee el departamento de Bolivar")

# %%
# Vemos cuales son las categorias que poseemos y cuantas son en total
categoria=df_bolivar["CATEGORIA"].unique()
df_bolivar["CATEGORIA"].unique()
#%%
print(f"La cantidad de categoria es de {len(categoria)}")

# %%
#Agrupamos los municipios y la cantidad de categorias se presenta en la municipio
df_bolivar.groupby("MUNICIPIO")["CATEGORIA"].nunique()
# %%
# Lo visualizamos mucho mejor con un mapa de calor
contingency_table = pd.crosstab(df_bolivar['MUNICIPIO'], df_bolivar['CATEGORIA'])
plt.figure(figsize=(18, 12))
# Usamos LogNorm para que Cartagena no domine toda la escala de colores
# y podamos ver la distribución en otros municipios.
sns.heatmap(contingency_table, 
            annot=True, 
            fmt='d', 
            cmap='viridis', 
            linewidths=.5,
            norm=LogNorm())
plt.title('Mapa de Calor: Concentración de Establecimientos por Municipio y Categoría', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Categoría del Servicio Turístico', fontsize=14)
plt.ylabel('Municipio', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Conclusiones del mapa de calor
#Hay 3 municipios de con mayor crecimiento de en el sector turistico son Cartagena, Turbaco y Arjona
#Se presenta que en los municipios de Hatillo de loba, Santa rosa, Soplaviento, talaigua nuevo, turbana, villanueva y zambrano, en el años de 2025 no presentaron un registro turistico en la categoria de establecimiento de alogamiento turistico
#En mompox es segundo municipio que presenta mayor agencias de viajes, establecimiento de alojamiento turistico y vivienda turistica por debajo de Cartagena
#En el caso de Achi presento un unico registro en el categoria de guias de turismo, a su vez, Villanueva presento una solo registro en la categoria de parque tematicos siendo los unicos municipios con un unico registro fuera de las categirias de alogamientos o similares y agenacias de viajes

#%%
# Clasificamos los municipios por 5 zonas turisticas, aclaron que apartamos la capital (cartagena) debido a que posee mayor concentraccion de turismo 
def assign_zone(municipio):
    """
    Asigna una zona turística basada en el municipio.
    Esta versión es más robusta porque:
    1. Maneja valores que no son texto (nulos).
    2. Convierte a mayúsculas para evitar problemas de mayúsculas/minúsculas.
    3. Usa 'in' para buscar subcadenas (ej. 'CARTAGENA' coincide con 'CARTAGENA DE INDIAS').
    """
    if not isinstance(municipio, str):
        return '5. Otros Municipios'  
    municipio_norm = municipio.strip().upper() 
    if 'TURBACO' in municipio_norm or 'SANTA CATALINA' in municipio_norm or 'VILLANUEVA' in municipio_norm or 'SANTA ROSA' in municipio_norm:
        return '2. Costa Caribe'
    elif 'ARENAL' in municipio_norm or 'MOMPOS' in municipio_norm or 'MAGANGUE' in municipio_norm or 'TALAIGUA NUEVO' in municipio_norm or 'ARJONA' in municipio_norm or 'HATILLO DE LOBA' in municipio_norm or 'MAHATES' in municipio_norm or 'MORALES' in municipio_norm or 'SAN ESTANISLAO' in municipio_norm or 'SAN PABLO' in municipio_norm or 'SOPLAVIENTO' in municipio_norm or 'TURBANA' in municipio_norm:
        return '3. Histórico-Fluvial'
    elif 'CARMEN DE BOLIVAR' in municipio_norm or 'SAN JACINTO' in municipio_norm or 'SAN JUAN NEPOMUCENO' in municipio_norm  or 'MARIA LA BAJA' in municipio_norm or 'ZAMBRANO' in municipio_norm: # Se corrigió 'MARÍA LA BAJA' para que coincida con la lista original
        return '4. Montes de María'
    elif 'CARTAGENA' in municipio_norm:
        return '1. Capital'
    else:
        return '5. Otros Municipios'
df_bolivar['ZONA_TURISTICA'] = df_bolivar['MUNICIPIO'].apply(assign_zone)
# Ver cuantos municipios se presenta en la zona turistica
df_bolivar.groupby("ZONA_TURISTICA")["MUNICIPIO"].nunique()
#%%
#Visualizar los municipios por zona horaria
pd.set_option('display.max_colwidth', None)
df_fila_unica = df_bolivar.groupby('ZONA_TURISTICA')['MUNICIPIO'].apply(
    lambda municipios: ', '.join(sorted(municipios.unique()))
).reset_index()
df_fila_unica.rename(columns={'MUNICIPIO': 'Municipios Incluidos'}, inplace=True)
df_fila_unica
#%%
#Ver como se comporta la zonas turistica respecto a las 5 categoria mas cotizadas 
top_5_categorias = df_bolivar['CATEGORIA'].value_counts().nlargest(5).index
df_zona_filtered = df_bolivar[df_bolivar['CATEGORIA'].isin(top_5_categorias)]
zone_composition = pd.crosstab(df_zona_filtered['ZONA_TURISTICA'], df_zona_filtered['CATEGORIA'])
zone_composition_percent = zone_composition.div(zone_composition.sum(axis=1), axis=0) * 100
ax = zone_composition_percent.plot(kind='barh', stacked=True, figsize=(16, 10), colormap='tab20c')
plt.title('Composición Porcentual de la Oferta Turística por Zona Geográfica', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Porcentaje de Establecimientos (%)', fontsize=14)
plt.ylabel('Zona Turística', fontsize=14)
plt.legend(title='Categoría', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# %%
#Visualizar cuantos empleados hay en cada zona turistica
zona_emp=df_bolivar.groupby('ZONA_TURISTICA')['NUM_EMP'].sum()
plt.figure(figsize=(10, 8))
plt.barh(zona_emp.index, zona_emp.values, color='coral')
plt.title('Número de Empleados por Zona Turística', fontsize=16)
plt.xlabel('Cantidad de Empleados', fontsize=12)
plt.ylabel('Zona Turística', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#%%
zona_emp

#%%
no_tiene_empleados=df_bolivar.query('NUM_EMP==0')
micro_empresa=df_bolivar.query('NUM_EMP>0 & NUM_EMP <=10')
pequena_empresa=df_bolivar.query('NUM_EMP>10 & NUM_EMP <=50')
mediana_empresa=df_bolivar.query('NUM_EMP>50 & NUM_EMP <=200')
gran_empresa=df_bolivar.query('NUM_EMP >200')

# Datos y etiquetas para el gráfico
sizes = [len(no_tiene_empleados), len(micro_empresa), len(pequena_empresa), len(mediana_empresa), len(gran_empresa)]
labels = ['Sin Empleados', 'Microempresa (1-10)', 'Pequeña (11-50)', 'Mediana (51-200)', 'Gran Empresa (>200)']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']

fig, ax = plt.subplots(figsize=(10, 6))

# Crear las barras
bars = ax.bar(labels, sizes, color=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])

# Añadir el título y las etiquetas
ax.set_title('Cantidad de Empresas por Tamaño', fontsize=16)
ax.set_ylabel('Número de Empresas', fontsize=12)
ax.set_xlabel('Categoría de Empresa', fontsize=12)

# Añadir el número exacto encima de cada barra
ax.bar_label(bars, padding=3)

# Quitar el borde superior y derecho para un look más limpio
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%
data_to_plot = [
    micro_empresa['NUM_EMP'],
    pequena_empresa['NUM_EMP'],
    mediana_empresa['NUM_EMP'],
    gran_empresa['NUM_EMP']
]

# Etiquetas para cada boxplot
labels = ['Micro', 'Pequeña', 'Mediana', 'Grande']

# 4. CREACIÓN DEL GRÁFICO
fig, ax = plt.subplots(figsize=(10, 7))

# Creación de los boxplots para cada categoría
# patch_artist=True permite rellenar las cajas con color
bplot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

# 5. TÍTULOS, ETIQUETAS Y ESCALA LOGARÍTMICA
ax.set_title('Distribución de Empleados por Categoría de Empresa', fontsize=16)
ax.set_ylabel('Número de Empleados (Escala Logarítmica)', fontsize=12)
ax.set_xlabel('Tipo de Empresa', fontsize=12)

# ---- PASO CLAVE: Se aplica la escala logarítmica ----
ax.set_yscale('log')

# Se añade una cuadrícula para facilitar la lectura en la escala logarítmica
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Opcional: Colorear las cajas para que se vean mejor
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# %%
#Agrupamos los meses con la cantidad de empleados 
empleo_por_mes=df_bolivar.groupby('MES')['NUM_EMP'].sum()
empleo_por_mes
# %%
# Vemos el comportamiento general de bolivar entre enero y mayo
nombres_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo']
plt.figure(figsize=(18, 10))
sns.lineplot(x=nombres_meses, y=empleo_por_mes.values, marker='o', linestyle='-', color='b', lw=2)
plt.title('Evolución del Empleo Durante el Periodo de Renovación del RNT (Ene-May)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Número Total de Empleados Registrados', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
top_3_categorias = df_bolivar['CATEGORIA'].value_counts().nlargest(5).index
df_top_categorias = df_bolivar[df_bolivar['CATEGORIA'].isin(top_3_categorias)]
empleo_categoria_mes = df_top_categorias.pivot_table(
    index='MES',
    columns='CATEGORIA',
    values='NUM_EMP',
    aggfunc='sum'
).reindex(range(1, 6), fill_value=0)
plt.figure(figsize=(18, 10))
# Graficar una línea para cada categoría
for categoria in empleo_categoria_mes.columns:
    sns.lineplot(x=nombres_meses, y=empleo_categoria_mes[categoria], marker='o', linestyle='--', label=categoria)  
plt.title('Evolución del Empleo por Categoría Durante el Periodo de Renovación (Ene-May)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Número Total de Empleados Registrados', fontsize=14)
plt.legend(title='Categoría Turística', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#%%
df_bolivar.groupby('MUNICIPIO')['NUM_EMP'].sum().sort_values()
# %%
pd.set_option('display.max_colwidth', None)
#Vamos a visualizar por cada municipio
data_agrupada_municipio=df_bolivar.groupby(['MES', 'MUNICIPIO'])['NUM_EMP'].sum()
df_pivot = data_agrupada_municipio.unstack('MUNICIPIO', fill_value=0)
df_pivot
# %%
df_pivot.plot(kind='line', figsize=(14, 7))
plt.title('Tendencia del Empleo por Municipio a lo largo del Año')
plt.xlabel('Mes')
plt.ylabel('Número de Empleados')
plt.grid(True) # Añade una cuadrícula para fácil lectura
plt.legend(title='Municipio')
plt.show()
# Se envidencia que por parte de en el municipio de cartagena contiene una gran parte de contratacion en comparacion de los otros municipios
# %%
data_agrupada_zonaTuristica=df_bolivar.groupby(['MES', 'ZONA_TURISTICA'])['NUM_EMP'].sum()
df_pivot2 = data_agrupada_zonaTuristica.unstack('ZONA_TURISTICA', fill_value=0)
df_pivot2
#%%
df_pivot2.plot(kind='line', figsize=(14, 7))
plt.title('Tendencia del Empleo por Zona Turistica a lo largo del Año')
plt.xlabel('Mes')
plt.ylabel('Número de Empleados')
plt.grid(True) # Añade una cuadrícula para fácil lectura
plt.legend(title='Zona Turistica')
plt.show()
#A su vez pasa los mismo en la zonas turistica sabiendo que cartagena es la capital

# %%
# Se estudia el comportamiento de las demas zonas 
zona_a_excluir = '1. Capital' # <-- Cambia este valor
df_filtrado_previo = df_bolivar[df_bolivar['ZONA_TURISTICA'] != zona_a_excluir]
data_agrupada = df_filtrado_previo.groupby(['MES', 'ZONA_TURISTICA'])['NUM_EMP'].sum()
df_pivot_final2 = data_agrupada.unstack('ZONA_TURISTICA', fill_value=0)
df_pivot_final2
#%%
df_pivot_final2.plot(kind='line', figsize=(14, 7))
plt.title('Tendencia del Empleo por Zona Turistica a lo largo del Año')
plt.xlabel('Mes')
plt.ylabel('Número de Empleados')
plt.grid(True) # Añade una cuadrícula para fácil lectura
plt.legend(title='Zona Turistica')
plt.show()
# Con respcto dados se conoce que en la zona que presenta mas empleo es la zona historico-flubial y denotado mas contrataciones en marzo, se puede decir que hacen contrataciones previos a la semana santa 
# muy representativa para esta zona, teniendo encuenta que el municipio de mompox esta en su mayor apogeo en esas fechas religiosas
 #%%
# Aca se ve que se presenta que hay municipios no tienen empleados por lo cual se puede deducir que esto establecimientos estan apenas empezando como tambien no hay mas empleado que se le da un salario minimo
# los municipios que no presentan empleados son HATILLO DE LOBA, TALAIGUA NUEVO y ZAMBRANO
sumas_mensuales = df_bolivar.groupby(['MUNICIPIO'])['NUM_EMP'].sum()
municipios_con_cero = sumas_mensuales[sumas_mensuales == 0]
municipios_con_cero
# %%
# Vamos estudiar mas a fondo sin contar en este caso a cartagena tanto en el municipio y ni la capital para las zonas turisticas
municipio_a_excluir = 'CARTAGENA'
df_filtrado_previo = df_bolivar[
    (df_bolivar['MUNICIPIO'] != municipio_a_excluir) &
    (~df_bolivar['MUNICIPIO'].isin(municipios_con_cero.index))
]
data_agrupada_con_zona = df_filtrado_previo.groupby(['MES', 'ZONA_TURISTICA', 'MUNICIPIO'])['NUM_EMP'].sum()
df_pivot_con_zonas = data_agrupada_con_zona.unstack(['ZONA_TURISTICA', 'MUNICIPIO'], fill_value=0)
df_final_ordenado = df_pivot_con_zonas.sort_index(axis=1)
df_final_ordenado
#%%
# Filtramos las columnas para quedarnos solo con la zona de interés y las graficamos
df_montes_de_maria = df_final_ordenado['2. Costa Caribe']
df_montes_de_maria.plot(
    kind='line',
    figsize=(14, 7),
    title='Evolución del Empleo en Costa Caribe (2025)',
    grid=True
)
plt.ylabel('Número de Empleados')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.legend(title='Municipios')
plt.tight_layout()
plt.show()
df_montes_de_maria = df_final_ordenado['3. Histórico-Fluvial']
df_montes_de_maria.plot(
    kind='line',
    figsize=(14, 7),
    title='Evolución del Empleo en Histórico-Fluvial (2025)',
    grid=True
)
plt.ylabel('Número de Empleados')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.legend(title='Municipios')
plt.tight_layout()
plt.show()
df_montes_de_maria = df_final_ordenado['4. Montes de María']
df_montes_de_maria.plot(
    kind='line',
    figsize=(14, 7), 
    title='Evolución del Empleo en Montes de María (2025)',
    grid=True
)
plt.ylabel('Número de Empleados')
plt.xlabel('Mes')
plt.xticks(rotation=45) 
plt.legend(title='Municipios')
plt.tight_layout()
plt.show()
df_montes_de_maria = df_final_ordenado['5. Otros Municipios']
df_montes_de_maria.plot(
    kind='line',
    figsize=(14, 7),
    title='Evolución del Empleo en Otros Municipios (2025)',
    grid=True
)
plt.ylabel('Número de Empleados')
plt.xlabel('Mes')
plt.xticks(rotation=45) 
plt.legend(title='Municipios')
plt.tight_layout() 
plt.show()
# %%
