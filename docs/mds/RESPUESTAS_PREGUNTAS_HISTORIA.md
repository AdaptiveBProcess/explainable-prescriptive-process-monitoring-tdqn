# 📋 Respuestas a las Preguntas sobre la Historia

## 1. ¿Por qué es peligroso practicar con personas reales?

**Respuesta:**

Cuando decimos "practicar con personas reales", nos referimos a que el agente RL aprenda **en vivo** usando datos del día a día de hoy, tomando decisiones reales sobre préstamos de clientes reales mientras está aprendiendo.

**¿Por qué es peligroso?**

1. **Riesgo financiero**: Si el agente toma malas decisiones mientras aprende, puede:
   - Aprobar préstamos que deberían rechazarse → pérdidas financieras
   - Rechazar préstamos buenos → pérdida de oportunidades
   - Contactar HQ innecesariamente → costos operativos

2. **Riesgo ético**: No podemos usar clientes reales como "conejillos de indias" para experimentar con políticas que aún no están validadas.

3. **Riesgo regulatorio**: Los bancos tienen regulaciones estrictas sobre cómo se toman decisiones crediticias. Un agente que aprende en vivo podría violar estas regulaciones.

**Por eso usamos "Offline RL":**
- El agente aprende de **datos históricos** (eventos pasados que ya ocurrieron)
- No toma decisiones reales mientras aprende
- Solo después de validar que funciona bien (con OPE), se despliega

**Código relacionado:**
- `src/xppm/rl/train_tdqn.py` - Entrenamiento offline (no interacción en vivo)
- `src/xppm/ope/doubly_robust.py` - Evaluación sin interacción (OPE)

---

## 2. ¿Qué parte del código hace decisiones múltiples y dónde se guardan?

**Respuesta:**

El robot toma **múltiples decisiones a lo largo del tiempo** para cada caso (préstamo). Esto se implementa en:

### Código que implementa decisiones múltiples:

1. **Construcción del dataset MDP** (`src/xppm/data/build_mdp.py`):
   - Líneas 269-433: `build_transitions()` crea múltiples transiciones por caso
   - Cada prefijo (momento en el tiempo) es un **punto de decisión**
   - Para cada caso, se generan transiciones en múltiples pasos temporales (`t=1, t=2, ..., t=L`)

2. **Entrenamiento TDQN** (`src/xppm/rl/train_tdqn.py`):
   - El modelo aprende `Q(s_t, a)` para cada estado `s_t` en la secuencia
   - En cada paso temporal, el modelo puede decidir una acción diferente
   - La política aprendida es **secuencial**: decisiones en t=1 afectan decisiones en t=2, t=3, etc.

3. **Explicaciones** (`src/xppm/xai/explain_policy.py`):
   - Selecciona múltiples transiciones por caso (líneas 324-339)
   - Genera explicaciones para cada momento de decisión
   - Puede explicar por qué decidió X en t=5 y Y en t=10

### Dónde se guardan los resultados:

**Dataset MDP:**
- `data/processed/D_offline.npz` - Contiene todas las transiciones (s, a, r, s_next)
- Cada transición tiene `case_ptr` (ID del caso) y `t_ptr` (paso temporal)

**Explicaciones por transición:**
- `artifacts/xai/final/risk_explanations.json` - Explicaciones de riesgo por transición
- `artifacts/xai/final/deltaQ_explanations.json` - Explicaciones contrastivas por transición
- Cada item tiene `case_id` y `t` (paso temporal)

**Ejemplo de estructura:**
```json
{
  "items": [
    {
      "case_id": 129,
      "t": 10,  // Decisión en paso 10
      "a_star": 0,
      "a_star_name": "do_nothing",
      "V": -264.77,
      "top_tokens": [...]
    },
    {
      "case_id": 129,
      "t": 11,  // Decisión en paso 11 (mismo caso, momento diferente)
      "a_star": 1,
      "a_star_name": "contact_headquarters",
      "V": 532.53,
      "top_tokens": [...]
    }
  ]
}
```

---

## 3. Explicación detallada de "Doubly Robust"

**¿Qué significa "dos formas de contar"?**

Doubly Robust combina **dos métodos diferentes** para estimar el valor de una política:

### Método 1: Direct Modeling (Modelado Directo)
- **Qué cuenta**: Predice directamente `Q(s,a)` usando el modelo entrenado
- **Cómo**: Usa la red neuronal TDQN para predecir `Q(s_t, a_t)` para cada transición
- **Ventaja**: Si el modelo es bueno, es muy preciso
- **Desventaja**: Si el modelo es malo, el error es grande

**Fórmula**: `V_π ≈ promedio de Q(s,a)` sobre todas las transiciones

### Método 2: Importance Sampling (Muestreo por Importancia)
- **Qué cuenta**: Pesa las recompensas observadas según qué tan probable es que la nueva política hubiera tomado esa acción vs la política antigua
- **Cómo**: Calcula `ρ = π_nueva(a|s) / π_antigua(a|s)` y multiplica las recompensas por este peso
- **Ventaja**: Funciona bien incluso si el modelo Q es malo
- **Desventaja**: Puede tener alta varianza si las políticas son muy diferentes

**Fórmula**: `V_π ≈ promedio de ρ * r` donde `ρ` es el peso de importancia

### Doubly Robust: Combinación de ambos
- **Qué cuenta**: Combina ambos métodos de forma inteligente
- **Cómo**: `DR = ρ * (r - Q(s,a)) + Q(s,a)`
  - Si el modelo Q es bueno → el término `(r - Q(s,a))` es pequeño → DR ≈ Q(s,a) (usa método 1)
  - Si el modelo Q es malo pero ρ es bueno → DR ≈ ρ * r (usa método 2)
  - **Es "robusto" porque funciona bien incluso si uno de los dos métodos falla**

**Código:**
- `src/xppm/ope/doubly_robust.py`, líneas 238-240:
```python
# Step-wise DR estimator:
#   DR_t = ρ_t * (r_t - Q(s_t, a_t)) + V(s_t)
dr_step = rho_trunc * (r - q_sa) + v_s
```

**Resultados guardados en:**
- `artifacts/ope/ope_dr.json` - Contiene `tdqn_dr_mean`, `tdqn_dr_ci95`, etc.

---

## 4. Comparación: Figura de Arquitectura vs Implementación Real

**Análisis de qué está implementado y qué no:**

### ✅ IMPLEMENTADO:

1. **Phase 1 - Data → Offline RLSet:**
   - ✅ `01_preprocess_log.py` → `clean.parquet`
   - ✅ `02_encode_prefixes.py` → `prefixes.npz`
   - ✅ `03_build_mdp_dataset.py` → `D_offline.npz`
   - ✅ `01b_validate_and_split.py` → `splits.json`

2. **Phase 2 - Training + OPE:**
   - ✅ `04_train_tdqn_offline.py` → `Q_theta.ckpt`
   - ✅ `05_run_ope_dr.py` → `ope_dr.json`
   - ✅ Behavior policy estimation
   - ✅ Doubly Robust estimator con bootstrap CIs

3. **Phase 3 - XAI:**
   - ✅ `06_explain_policy.py` → Risk + DeltaQ explanations
   - ✅ Integrated Gradients attributions
   - ✅ Policy summary (clustering)
   - ✅ `07_fidelity_tests.py` → Q-drop, Action-flip, Rank-consistency
   - ✅ `08_distill_policy.py` → Decision tree surrogate

4. **Deployment:**
   - ✅ `policy_server.py` (FastAPI)
   - ✅ Policy Guard (OOD detection, uncertainty threshold)
   - ✅ Decision logging

### ⚠️ PARCIALMENTE IMPLEMENTADO:

1. **Counterfactual Rollouts:**
   - ❌ Mencionado en la figura (línea 199) pero **NO implementado**
   - Los fidelity tests solo hacen Q-drop y action-flip, no rollouts completos

2. **Monitoring:**
   - ✅ Scripts de monitoreo existen (`13_compute_monitoring_metrics.py`, `14_detect_drift.py`)
   - ⚠️ Pero el feedback loop completo (línea 313 de la figura) está parcialmente implementado

### ❌ NO IMPLEMENTADO:

1. **Experiment Tracking:**
   - La figura muestra W&B/MLflow (líneas 48-52)
   - ⚠️ Configurado pero no siempre usado en todos los scripts

2. **CI/Tests:**
   - La figura muestra pytest (líneas 55-59)
   - ✅ Tests existen pero no todos los componentes están cubiertos

**Voy a crear una figura actualizada que refleje el estado real:**

(Ver archivo `figure-arquitecture-actual.tex`)

---

## 5. ¿Dónde está el código que explica el plan completo?

**Respuesta:**

El código que genera las explicaciones del "plan completo" (no solo una decisión aislada) está en:

### Código principal:

**`src/xppm/xai/explain_policy.py`** (función `explain_policy()`, líneas 255-616):
- **Líneas 324-339**: Selecciona múltiples transiciones por caso (`k_times_per_case`)
- **Líneas 350-450**: Genera **Risk explanations** (por qué el caso es riesgoso)
- **Líneas 450-550**: Genera **DeltaQ explanations** (por qué esta ayuda es mejor que otra)
- **Líneas 550-600**: Genera **Policy summary** (clustering de estrategias)

**`src/xppm/xai/attributions.py`**:
- **Líneas 81-150**: `integrated_gradients_embedding()` - Calcula atribuciones IG
- **Líneas 150-250**: `compute_attributions()` - Wrapper que calcula atribuciones para múltiples targets

**`src/xppm/xai/policy_summary.py`**:
- **Líneas 18-62**: `extract_encoder_embeddings()` - Extrae representaciones de estados
- **Líneas 100-250**: `summarize_policy()` - Agrupa estados similares en clusters/estrategias

### Dónde se guardan los resultados:

**Risk Explanations** (`artifacts/xai/final/risk_explanations.json`):
- Cada item explica **por qué el caso es riesgoso** en un momento específico
- Contiene `V(s_t)` (valor del estado) y `top_tokens` (tokens más importantes)
- Ejemplo:
```json
{
  "case_id": 129,
  "t": 10,
  "V": -264.77,  // Valor bajo = caso riesgoso
  "top_tokens": [
    {"position": 49, "token_name": "skip_contact", "importance": 7386.17}
  ]
}
```

**DeltaQ Explanations** (`artifacts/xai/final/deltaQ_explanations.json`):
- Cada item explica **por qué una acción es mejor que otra**
- Contiene `delta_q = Q(a*) - Q(a_contrast)` y `top_drivers` (drivers de la diferencia)
- Ejemplo:
```json
{
  "case_id": 552,
  "t": 10,
  "a_star": 1,  // contact_headquarters
  "a_contrast": 0,  // do_nothing
  "delta_q": 797.30,  // Gran diferencia = intervención mucho mejor
  "top_drivers": [
    {"position": 48, "token_name": "skip_contact", "importance": 75.11}
  ]
}
```

**Policy Summary** (`artifacts/xai/final/policy_summary.json`):
- Agrupa estados similares en **clusters** (estrategias)
- Cada cluster tiene:
  - `action_distribution`: Qué acciones toma en este tipo de estados
  - `mean_v`: Valor promedio del cluster
  - `mean_delta_q`: Diferencia promedio entre acciones
  - `prototypes`: Ejemplos representativos del cluster
- Ejemplo:
```json
{
  "cluster_id": 1,
  "n": 33558,
  "action_distribution": {
    "do_nothing": 0.0,
    "contact_headquarters": 1.0
  },
  "mean_v": 1572.46,
  "mean_delta_q": 934.48,
  "prototypes": [
    {"case_id": 37930, "t": 8, "v": 1601.01}
  ]
}
```

### Cómo se explica el "plan completo":

1. **Múltiples momentos**: Las explicaciones se generan para múltiples pasos temporales (`t=1, t=2, ..., t=L`) del mismo caso
2. **Secuencia de decisiones**: Cada explicación muestra por qué se decidió X en el momento t
3. **Estrategia agregada**: El policy summary agrupa estados similares para mostrar "patrones de decisión" (estrategias)

---

## 6. ¿Qué significa "necesitamos muchas formas diferentes de probar las explicaciones"?

**Respuesta:**

Esta frase se refiere a que **una sola prueba de fidelidad no es suficiente** para validar que las explicaciones son confiables. Necesitamos múltiples pruebas que validen diferentes aspectos.

### ¿Qué pruebas de fidelidad tenemos actualmente?

**✅ IMPLEMENTADAS:**

1. **Q-drop** (`src/xppm/xai/fidelity_tests.py`, función `_test_q_drop`, líneas 253-525):
   - **Qué prueba**: Si quitamos tokens importantes, ¿baja más el valor Q que si quitamos tokens aleatorios?
   - **Qué valida**: Que las explicaciones realmente identifican qué tokens afectan el valor Q
   - **Resultado**: Gap positivo = explicaciones son útiles

2. **Action-flip** (`src/xppm/xai/fidelity_tests.py`, función `_test_action_flip`, líneas 528-819):
   - **Qué prueba**: Si quitamos tokens importantes, ¿cambia más la acción que si quitamos tokens aleatorios?
   - **Qué valida**: Que las explicaciones identifican qué tokens afectan la decisión
   - **Resultado**: Flip rate mayor con top-k = explicaciones son útiles

3. **Rank-consistency** (`src/xppm/xai/fidelity_tests.py`, función `_test_rank_consistency`, líneas 822-945):
   - **Qué prueba**: ¿El ranking por Q coincide con el ranking por OPE (proxy)?
   - **Qué valida**: Que las explicaciones son consistentes con métricas globales
   - **Resultado**: Correlación Spearman/Kendall alta = consistencia

**❌ NO IMPLEMENTADAS (pero mencionadas en la literatura):**

4. **Counterfactual Rollouts**:
   - **Qué probaría**: Si seguimos la explicación y cambiamos los tokens importantes, ¿qué pasa en el futuro?
   - **Qué validaría**: Que las explicaciones predicen efectos reales a largo plazo
   - **Estado**: Mencionado en la figura pero NO implementado

5. **Sensitivity Analysis**:
   - **Qué probaría**: ¿Qué tan robustas son las explicaciones a pequeños cambios en los inputs?
   - **Qué validaría**: Estabilidad de las explicaciones

6. **Human Evaluation**:
   - **Qué probaría**: ¿Los humanos entienden y confían en las explicaciones?
   - **Qué validaría**: Utilidad práctica de las explicaciones

### ¿Tenemos todas las pruebas necesarias?

**Respuesta corta: NO, pero tenemos las más importantes.**

**Lo que tenemos (3 pruebas):**
- ✅ Q-drop: Valida que las explicaciones afectan el valor Q
- ✅ Action-flip: Valida que las explicaciones afectan las decisiones
- ✅ Rank-consistency: Valida consistencia global

**Lo que falta (pero sería deseable):**
- ❌ Counterfactual rollouts: Validación más fuerte de efectos causales
- ❌ Sensitivity analysis: Validación de robustez
- ❌ Human evaluation: Validación de utilidad práctica

**Conclusión:** Tenemos las pruebas **mínimas necesarias** para publicar, pero idealmente deberíamos agregar más pruebas para hacer el paper más fuerte.

**Resultados guardados en:**
- `artifacts/fidelity/fidelity.csv` - Contiene todas las métricas de las 3 pruebas

---

## 7. ¿Dónde están las explicaciones para leerlas?

**Respuesta:**

Las explicaciones están guardadas en archivos JSON en:

### Ubicaciones principales:

1. **`artifacts/xai/final/risk_explanations.json`**
   - Explicaciones de riesgo (por qué el caso es riesgoso)
   - Formato: JSON con array de items, cada item tiene:
     - `case_id`: ID del caso
     - `t`: Paso temporal
     - `V`: Valor del estado (bajo = riesgoso)
     - `top_tokens`: Lista de tokens más importantes con su importancia

2. **`artifacts/xai/final/deltaQ_explanations.json`**
   - Explicaciones contrastivas (por qué una acción es mejor que otra)
   - Formato: JSON con array de items, cada item tiene:
     - `case_id`, `t`: Identificación
     - `a_star`: Acción recomendada
     - `a_contrast`: Acción de contraste
     - `delta_q`: Diferencia en Q-values
     - `top_drivers`: Tokens que explican la diferencia

3. **`artifacts/xai/final/policy_summary.json`**
   - Resumen de la política (clusters/estrategias)
   - Formato: JSON con:
     - `clusters`: Array de clusters, cada uno con:
       - `cluster_id`: ID del cluster
       - `n`: Número de estados en el cluster
       - `action_distribution`: Distribución de acciones
       - `mean_v`, `mean_delta_q`: Métricas del cluster
       - `prototypes`: Ejemplos representativos

### Cómo leerlas:

**Opción 1: Ver directamente con `cat` o editor de texto:**
```bash
cat artifacts/xai/final/risk_explanations.json | python -m json.tool | less
```

**Opción 2: Usar Python para explorar:**
```python
import json

# Leer risk explanations
with open('artifacts/xai/final/risk_explanations.json') as f:
    risk = json.load(f)

# Ver primer caso
print("Primer caso:")
print(f"Case ID: {risk['items'][0]['case_id']}")
print(f"Paso temporal: {risk['items'][0]['t']}")
print(f"Valor V: {risk['items'][0]['V']}")
print(f"Top tokens importantes:")
for token in risk['items'][0]['top_tokens'][:5]:
    print(f"  - {token['token_name']} (posición {token['position']}): importancia {token['importance']:.2f}")
```

**Opción 3: Ver en el bundle de deployment:**
- `artifacts/deploy/v1/xai/risk_explanations.json`
- `artifacts/deploy/v1/xai/deltaQ_explanations.json`
- `artifacts/deploy/v1/xai/policy_summary.json`

### Ejemplo de lectura:

**Risk Explanation (caso 129, t=10):**
- **Caso**: 129
- **Momento**: Paso 10
- **Valor V**: -264.77 (negativo = caso riesgoso)
- **Tokens más importantes**:
  1. `skip_contact` (posición 49): importancia 7386.17
  2. `email_customer` (posición 47): importancia 321.40
  3. `start_standard` (posición 41): importancia 241.43

**Interpretación**: El caso es riesgoso principalmente porque tiene `skip_contact` al final de la secuencia.

**DeltaQ Explanation (caso 552, t=10):**
- **Caso**: 552
- **Momento**: Paso 10
- **Acción recomendada**: `contact_headquarters` (Q = 532.53)
- **Acción de contraste**: `do_nothing` (Q = -264.77)
- **Delta Q**: 797.30 (gran diferencia = intervención mucho mejor)
- **Drivers principales**:
  1. `skip_contact` (posición 48): importancia 75.11
  2. `validate_application` (posición 49): importancia 35.81

**Interpretación**: Contactar HQ es mucho mejor que no hacer nada, principalmente porque el caso tiene `skip_contact` y múltiples `validate_application`.

---

## Resumen Final

1. **¿Por qué es peligroso practicar con personas reales?**
   - Porque implica tomar decisiones reales mientras se aprende → riesgo financiero, ético y regulatorio
   - Por eso usamos Offline RL (aprende de datos históricos)

2. **¿Qué parte del código hace decisiones múltiples?**
   - `src/xppm/data/build_mdp.py` crea múltiples transiciones por caso
   - `src/xppm/xai/explain_policy.py` genera explicaciones para múltiples momentos
   - Resultados en `artifacts/xai/final/*.json`

3. **¿Qué es Doubly Robust?**
   - Combina dos métodos: Direct Modeling (Q-values) + Importance Sampling (pesos)
   - Es "robusto" porque funciona bien incluso si uno de los métodos falla
   - Código en `src/xppm/ope/doubly_robust.py`

4. **¿Qué está implementado vs la figura?**
   - ✅ Phase 1, 2, 3 principales implementadas
   - ⚠️ Counterfactual rollouts NO implementados
   - ⚠️ Monitoring parcialmente implementado
   - Ver `figure-arquitecture-actual.tex` para detalles

5. **¿Dónde está el código que explica el plan completo?**
   - `src/xppm/xai/explain_policy.py` genera explicaciones para múltiples momentos
   - `src/xppm/xai/policy_summary.py` agrupa estrategias
   - Resultados en `artifacts/xai/final/*.json`

6. **¿Qué significa "necesitamos muchas formas diferentes de probar"?**
   - Significa que una sola prueba no es suficiente
   - Tenemos 3 pruebas (Q-drop, Action-flip, Rank-consistency)
   - Faltan algunas pruebas avanzadas (rollouts, sensitivity, human eval)

7. **¿Dónde están las explicaciones para leerlas?**
   - `artifacts/xai/final/risk_explanations.json`
   - `artifacts/xai/final/deltaQ_explanations.json`
   - `artifacts/xai/final/policy_summary.json`
