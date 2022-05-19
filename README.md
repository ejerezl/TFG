# TFG
En este repositorio se incluye todo el código desarrollado al elaborar el Trabajo de Fin de Grado "Análisis de posibles casos de hipoxia fetal intraparto".

Con el paso del tiempo, los volúmenes de datos generados son cada vez mayores, siendo de
especial interés la aplicación de técnicas de aprendizaje automático para analizarlos y sacar, a partir
de ellos, información de valor. De entre la multitud de campos de la sociedad que se benefician de
estas tecnologías, el campo de la medicina ha sido capaz de aprovechar el gran número de datos
clínicos disponibles aplicándolos a modelos de pronóstico y diagnóstico que ayudan al especialista.
La vigilancia del bienestar fetal durante el embarazo, y especialmente en el parto, es una de las
principales tareas que realizan los obstetras. Para desempeñar esta labor, los expertos se apoyan
en las cardiotocografías mediante su análisis visual para detectar posibles casos de hipoxia fetal. Sin
embargo, la subjetividad que involucra este proceso, dificulta la eficiencia en su detección. Con objetivo
de clarificar este análisis para los expertos, surge la idea de automatizar el proceso.
El objetivo principal de este repositorio es, mediante el análisis de cardiotocografías, avanzar en la
detección de hipoxia fetal intraparto, problema que a día de hoy sigue existiendo. Para abordar es-
ta tarea, que actualmente tiene una casuística bastante compleja, se estudiarán diferentes métodos
de clústering que forman parte del aprendizaje no supervisado. Más concretamente, se explicará el
funcionamiento de diferentes tipos de clústering jerárquico y del algoritmo de K-medias. Siguiendo la
naturaleza de los datos que se quieren estudiar, dedicaremos una sección para hablar de datos funcio-
nales. Como resultado, se plantea el análisis como un problema de clústering, dividiéndolo en análisis
funcional y multivariante. Partiendo de una selección de variables que ayudan en la detección de
hipoxia, se concluye que con los métodos aplicados los grupos de observaciones obtenidos para el
análisis funcional son muy homogéneos, mientras que en el caso multivariante se dejan ver distintas
estructuras entre los datos.

Con el anterior objetivo presentamos tres estudios diferentes:
1. Por una parte, estudio de métodos de clústering con datos sintéticos en el Módulo 'Test clústering con datos sintéticos'
2. También un estudio de datos funcionales para acomodarnos al problema futuro. En este caso en el Módulo 'Berkeley data' se hace un estudio de diferentes métricas con K-medias
3. Finalmente, pasamos al estudio de detección de hipoxia intrafetal en el Módulo 'Detección de hipoxia'
