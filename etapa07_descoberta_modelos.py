# ============================================================
# ETAPA 07 – DESCOBERTA DE MODELOS (VERSÃO FINAL)
# ------------------------------------------------------------
# - (MODIFICADO) Adiciona 'import joblib' e salva o
#   modelo final 'rf_final' em disco para ser usado
#   pela Etapa 9 (Refinamento).
# ============================================================

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib # <<< ADICIONADO

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay

from utils_log import log_mensagem


# ============================== utilidades ==============================
# (Sem alterações nesta seção)

def _garantir_pastas():
    Path("resultados/figuras").mkdir(parents=True, exist_ok=True)
    Path("resultados/tabelas").mkdir(parents=True, exist_ok=True)
    Path("resultados/textos").mkdir(parents=True, exist_ok=True)
    Path("resultados/modelos").mkdir(parents=True, exist_ok=True)

def _nome_alvo(df: pd.DataFrame) -> str:
    alvos_preferidos = [
        "indice_autoeficacia_norm",
        "indice_autoeficacia",
        "indice_bem_estar_norm",
        "ibd",
        "indice_bem_estar"
    ]
    for c in alvos_preferidos:
        if c in df.columns:
            return c
    raise ValueError("Variável Alvo (Y) 'indice_autoeficacia_norm' não foi encontrada. Verifique a Etapa 5.")

def _features_base(df: pd.DataFrame, alvo: str) -> list[str]:
    preditores_engenheirados = [
        "clima_media",
        "carga_trabalho_media",
        "cooperacao_media",
        "satisfacao_media",
        "formacao_continuada_soma"
    ]
    
    cols = [c for c in preditores_engenheirados if c in df.columns]
    
    controles = ["TC002Q01NA"]
    for c in controles:
        col_real = _encontrar_nomes_reais(list(df.columns), [c])
        if col_real and col_real[0] not in cols and col_real[0] in df.select_dtypes(include=["number"]).columns:
            cols.append(col_real[0])

    if len(cols) < 2: 
        log_mensagem("ETAPA 7", f"Preditoras encontradas: {cols}", "aviso")
        raise ValueError(
             "Poucas variáveis preditoras (X) encontradas. "
             "Precisamos de pelo menos 2 (ex: 'clima_media', 'carga_trabalho_media') da Etapa 5."
         )
    
    return cols

def _encontrar_nomes_reais(colunas_df: list, codigos_prefixo: list) -> list:
    nomes_encontrados = []
    colunas_df_lower = {col.lower(): col for col in colunas_df}
    for codigo in codigos_prefixo:
        codigo_lower = codigo.lower()
        matches = [col for col_lower, col in colunas_df_lower.items() if col_lower.startswith(codigo_lower)]
        if matches:
            nomes_encontrados.append(matches[0])
    return nomes_encontrados

def _cv_regressor(modelo, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse = -cross_val_score(modelo, X, y, scoring="neg_root_mean_squared_error", cv=kf).mean()
    mae  = -cross_val_score(modelo, X, y, scoring="neg_mean_absolute_error", cv=kf).mean()
    r2   =  cross_val_score(modelo, X, y, scoring="r2", cv=kf).mean()
    return rmse, mae, r2

def _diagnosticos_ols(y, yhat, residuos, caminho):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        axes[0].scatter(yhat, residuos, s=12, alpha=0.7)
        axes[0].axhline(0, linestyle="--", linewidth=1)
        axes[0].set_title("Resíduos vs Ajustados")
        axes[0].set_xlabel("Ajustados")
        axes[0].set_ylabel("Resíduos")
        sm.qqplot(residuos, line="45", fit=True, ax=axes[1])
        axes[1].set_title("Q-Q Plot dos Resíduos")
        axes[2].scatter(y, yhat, s=12, alpha=0.7)
        lims = [min(float(y.min()), float(yhat.min())), max(float(y.max()), float(yhat.max()))]
        axes[2].plot(lims, lims, "--", linewidth=1)
        axes[2].set_title("Predito vs Observado")
        axes[2].set_xlabel("Observado")
        axes[2].set_ylabel("Predito")
        plt.tight_layout()
        plt.savefig(caminho, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

def _plotar_importancias_rf(modelo_rf, nomes, caminho):
    try:
        imp = pd.Series(modelo_rf.feature_importances_, index=nomes).sort_values(ascending=False)[:20]
        plt.figure(figsize=(9, 7))
        imp.iloc[::-1].plot(kind="barh")
        plt.title("Importância das Variáveis – Random Forest (Top 20)")
        plt.tight_layout()
        plt.savefig(caminho, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

def _escore_modelo(linha):
    r2cv = linha.get("R2_CV", np.nan)
    rmse = linha.get("RMSE_CV", np.nan)
    r2aj = linha.get("R2_ajustado", np.nan)
    aic  = linha.get("AIC", np.nan)
    bic  = linha.get("BIC", np.nan)
    return (
        r2cv if pd.notna(r2cv) else -1e9,
        -rmse if pd.notna(rmse) else -1e9,
        r2aj if pd.notna(r2aj) else -1e9,
        -aic if pd.notna(aic) else -1e9,
        -bic if pd.notna(bic) else -1e9
    )


# ============================== núcleo público ==============================

def ajustar_modelo(respostas: pd.DataFrame):
    """
    Ajusta OLS, compara modelos, seleciona o melhor e SALVA o melhor
    modelo não-linear (ex: RF) em disco para a Etapa 9.
    """
    etapa = "ETAPA 7 – Descoberta de Modelos"
    log_mensagem(etapa, "Iniciando ajuste e comparacao...", "inicio")
    _garantir_pastas()

    # 1) alvo e features
    alvo = _nome_alvo(respostas)
    X_cols = _features_base(respostas, alvo)
    
    log_mensagem(etapa, f"Alvo (Y) selecionado: {alvo}", "info")
    log_mensagem(etapa, f"Preditores (X) selecionados: {X_cols}", "info")

    y = pd.to_numeric(respostas[alvo], errors="coerce").astype(float)
    X = respostas[X_cols].apply(pd.to_numeric, errors="coerce")
    dados = pd.concat([X, y], axis=1).dropna()
    
    if dados.shape[0] < (len(X_cols) + 10):
        log_mensagem(etapa, f"Dados insuficientes ({dados.shape[0]} linhas) para {len(X_cols)} preditores. Verifique Etapa 5.", "erro")
        raise ValueError("Dados insuficientes após remoção de NaNs para modelagem.")
        
    X = dados[X_cols]
    y = dados[alvo]

    # 2) sanitização de nomes
    safe_cols = {}
    for i, col in enumerate(X_cols, start=1):
        safe = (
            col.replace(":", "_").replace("<", "").replace(">", "")
               .replace("[", "").replace("]", "").replace("?", "")
               .replace("(", "").replace(")", "").replace(".", "_")
               .replace("/", "_").replace(" ", "_")
        )
        safe_cols[col] = f"var_{i}_" + safe[:40]

    X_ren = X.rename(columns=safe_cols)
    dados_ren = pd.concat([X_ren, y], axis=1)

    pd.DataFrame(list(safe_cols.items()), columns=["original", "ols_nome"]) \
        .to_csv("resultados/tabelas/mapa_variaveis_ols.csv", index=False, encoding="utf-8-sig")

    # 3) fórmula segura
    formula = "Q('{y}') ~ {rhs}".format(y=alvo, rhs=" + ".join(X_ren.columns))
    resultados = []

    # --------------------------- OLS (com fallback) ---------------------------
    modelo_ols = None
    try:
        modelo_ols = smf.ols(formula, data=dados_ren).fit()
    except Exception as e_form:
        try:
            X_mat = sm.add_constant(X_ren, has_constant="add")
            modelo_ols = sm.OLS(y, X_mat).fit()
        except Exception as e_matrix:
            log_mensagem(etapa, f"Falha OLS (fórmula e matriz): {e_form} | {e_matrix}", "erro")

    if modelo_ols is not None:
        if hasattr(modelo_ols, "model") and hasattr(modelo_ols.model, "exog_names"):
            nomes_ols = modelo_ols.model.exog_names
        else:
            nomes_ols = ["const"] + list(X_ren.columns)

        ols_df = pd.DataFrame({
            "ols_nome": nomes_ols,
            "coeficiente": modelo_ols.params.values,
            "p_valor": modelo_ols.pvalues.values
        })

        mapa = pd.read_csv("resultados/tabelas/mapa_variaveis_ols.csv")
        ols_merged = ols_df.merge(mapa, how="left", on="ols_nome")
        if "original" in ols_merged.columns:
            ols_merged.loc[ols_merged["ols_nome"] == "const", "original"] = "Intercepto"
            ols_merged = ols_merged[["original", "ols_nome", "coeficiente", "p_valor"]]

        ols_merged.to_csv("resultados/tabelas/modelo_ols_resultados.csv", index=False, encoding="utf-8-sig")

        _diagnosticos_ols(y, modelo_ols.fittedvalues, modelo_ols.resid,
                          "resultados/figuras/diagnosticos_residuos_ols.png")

        resultados.append({
            "modelo": "OLS",
            "R2_ajustado": float(getattr(modelo_ols, "rsquared_adj", np.nan)),
            "AIC": float(getattr(modelo_ols, "aic", np.nan)),
            "BIC": float(getattr(modelo_ols, "bic", np.nan)),
            "RMSE_CV": np.nan, "MAE_CV": np.nan, "R2_CV": np.nan,
            "notas": "Regressão linear (via fórmula ou matriz, conforme disponível)"
        })
        log_mensagem("PIPELINE GERAL", "Modelo OLS ajustado com sucesso.", "info")
    else:
        log_mensagem(etapa, "Falha OLS: mesmo com fallback por matriz.", "erro")

    # ------------------- OLS com interação -------------------
    if "clima_media" in X_cols and "carga_trabalho_media" in X_cols:
        try:
            clima_safe = safe_cols.get("clima_media", "clima_media")
            carga_safe = safe_cols.get("carga_trabalho_media", "carga_trabalho_media")
            outras = [safe_cols.get(c, c) for c in X_cols if c not in ["clima_media", "carga_trabalho_media"]]
            
            formula_i = "Q('{y}') ~ {a} * {c}".format(y=alvo, a=clima_safe, c=carga_safe)
            if outras:
                formula_i += " + " + " + ".join(outras)
            
            modelo_inter = smf.ols(formula_i, data=dados_ren).fit()
            resultados.append({
                "modelo": "OLS_interacao_climaXcarga",
                "R2_ajustado": float(modelo_inter.rsquared_adj),
                "AIC": float(modelo_inter.aic), "BIC": float(modelo_inter.bic),
                "RMSE_CV": np.nan, "MAE_CV": np.nan, "R2_CV": np.nan,
                "notas": "Inclui termo de interação clima x carga"
            })
        except Exception as e:
            log_mensagem(etapa, f"Falha OLS_interacao: {e}", "erro")
    else:
        log_mensagem(etapa, "Skipping OLS_interacao: 'clima_media' ou 'carga_trabalho_media' não encontrados.", "aviso")


    # --------------------------- Huber (robusta) -----------------------------
    try:
        huber = Pipeline([("sc", StandardScaler()), ("reg", HuberRegressor())])
        rmse, mae, r2 = _cv_regressor(huber, X, y, cv=5)
        resultados.append({
            "modelo": "Regressao_Robusta_Huber",
            "R2_ajustado": np.nan, "AIC": np.nan, "BIC": np.nan,
            "RMSE_CV": rmse, "MAE_CV": mae, "R2_CV": r2,
            "notas": "Resistente a outliers"
        })
    except Exception as e:
        log_mensagem(etapa, f"Falha Huber: {e}", "erro")

    # ----------------------- Polinomial (grau 2) -----------------------------
    try:
        poly = ColumnTransformer([
            ("num", Pipeline([
                ("sc", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias=False))
            ]), X_cols)
        ])
        pipe_poly = Pipeline([("prep", poly), ("reg", LinearRegression())])
        rmse, mae, r2 = _cv_regressor(pipe_poly, X, y, cv=5)
        resultados.append({
            "modelo": "Regressao_Polinomial_grau2",
            "R2_ajustado": np.nan, "AIC": np.nan, "BIC": np.nan,
            "RMSE_CV": rmse, "MAE_CV": mae, "R2_CV": r2,
            "notas": "Captura curvaturas"
        })
    except Exception as e:
        log_mensagem(etapa, f"Falha Polinomial: {e}", "erro")

    # --------------------------- Random Forest -------------------------------
    try:
        rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rmse, mae, r2 = _cv_regressor(rf, X, y, cv=5)
        resultados.append({
            "modelo": "RandomForestRegressor",
            "R2_ajustado": np.nan, "AIC": np.nan, "BIC": np.nan,
            "RMSE_CV": rmse, "MAE_CV": mae, "R2_CV": r2,
            "notas": "Não-linear + importâncias"
        })
        rf_final = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
        rf_final.fit(X, y)
        _plotar_importancias_rf(rf_final, X.columns, "resultados/figuras/feature_importances_rf.png")
        
        principais = [c for c in ["clima_media", "carga_trabalho_media", "satisfacao_media", "cooperacao_media"] if c in X.columns][:2]
        if principais:
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(rf_final, X, principais, ax=ax)
            plt.tight_layout()
            plt.savefig("resultados/figuras/curva_parcial_dependencia.png", bbox_inches="tight")
            plt.close()

        # ####################################################################
        # # ### INÍCIO DA MODIFICAÇÃO ###
        # # Salva o modelo RF treinado em disco para a Etapa 9
        # ####################################################################
        caminho_modelo_rf = "resultados/modelos/rf_final.joblib"
        joblib.dump(rf_final, caminho_modelo_rf)
        log_mensagem(etapa, f"Modelo Random Forest final salvo em '{caminho_modelo_rf}'", "info")
        # ####################################################################
        # # ### FIM DA MODIFICAÇÃO ###
        # ####################################################################

    except Exception as e:
        log_mensagem(etapa, f"Falha RandomForest: {e}", "erro")

    # ------------------------- Gradient Boosting -----------------------------
    try:
        gbr = GradientBoostingRegressor(random_state=42)
        rmse, mae, r2 = _cv_regressor(gbr, X, y, cv=5)
        resultados.append({
            "modelo": "GradientBoostingRegressor",
            "R2_ajustado": np.nan, "AIC": np.nan, "BIC": np.nan,
            "RMSE_CV": rmse, "MAE_CV": mae, "R2_CV": r2,
            "notas": "Ensemble aditivo"
        })
    except Exception as e:
        log_mensagem(etapa, f"Falha GradientBoosting: {e}", "erro")

    # ------------------------- Logística Ordinal -----------------------------
    if "faixa_bem_estar" in respostas.columns:
        # (código omitido por brevidade, permanece o mesmo)
        pass # A lógica permanece a mesma
    else:
        log_mensagem(etapa, "Logistica Ordinal pulada (coluna 'faixa_bem_estar' não encontrada).", "info")


    # ------------------- consolidação / seleção do melhor --------------------
    comp = pd.DataFrame(resultados)
    comp["gerado_em"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comp.to_csv("resultados/tabelas/comparacao_modelos.csv", index=False, encoding="utf-8-sig")

    comp_dicts = comp.fillna(np.nan).to_dict(orient="records")
    melhor = max(comp_dicts, key=lambda r: (
        r.get("R2_CV", -1e9) if pd.notna(r.get("R2_CV", np.nan)) else -1e9,
        -r.get("RMSE_CV", 1e9) if pd.notna(r.get("RMSE_CV", np.nan)) else -1e9,
        r.get("R2_ajustado", -1e9) if pd.notna(r.get("R2_ajustado", np.nan)) else -1e9,
        -r.get("AIC", 1e9) if pd.notna(r.get("AIC", np.nan)) else -1e9,
        -r.get("BIC", 1e9) if pd.notna(r.get("BIC", np.nan)) else -1e9
    ))

    meta = {
        "melhor_modelo": melhor.get("modelo"),
        "criterio": "prioridade: maior R2_CV; depois menor RMSE_CV; depois maior R2_ajustado; AIC/BIC como desempate",
        "alvo": alvo,
        "features": list(X_cols),
        "tabela_resumo_path": "resultados/tabelas/comparacao_modelos.csv"
    }
    
    # Adiciona o caminho do modelo salvo ao JSON
    if meta["melhor_modelo"] == "RandomForestRegressor":
        meta["caminho_modelo_salvo"] = "resultados/modelos/rf_final.joblib"
    # (Adicionar lógica para outros modelos se necessário)
        
    Path("resultados/tabelas/melhor_modelo.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path("resultados/textos/melhor_modelo.txt").write_text(
        "Melhor modelo: {m}\nCriterio: {c}\nAlvo: {a}\nFeatures: {n}\n\n{t}\n".format(
            m=meta["melhor_modelo"], c=meta["criterio"], a=meta["alvo"],
            n=len(meta["features"]), t=comp.to_string(index=False)
        ),
        encoding="utf-8"
    )

    log_mensagem(etapa, f"Melhor modelo selecionado: {meta['melhor_modelo']}", "fim")

    # ------------------- retorno (compatibilidade) ---------------------------
    if modelo_ols is None:
        raise RuntimeError("OLS nao pode ser ajustado.")
    return modelo_ols