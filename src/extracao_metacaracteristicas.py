import argparse
import os
import time
import warnings
from datetime import datetime

import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE

GRUPOS_MFE = ["general", "statistical", "info-theory"]
EXCLUIR_MFE = {"attr_conc.mean", "attr_conc.sd"}

ARQ_LOG = None


def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    if ARQ_LOG:
        with open(ARQ_LOG, "a", encoding="utf-8") as f:
            f.write(linha + "\n")


def salvar_csv_atomico(df, caminho):
    tmp = caminho + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, caminho)


def extrair_task(tid):
    task = openml.tasks.get_task(int(tid), download_data=False)
    dataset = task.get_dataset()
    X, y, cat_ind, _ = dataset.get_data(target=task.target_name)
    X = X.copy()

    #Fração global de valores faltantes
    frac_missing = float(X.isna().to_numpy().mean()) if X.size > 0 else 0.0

    #Descarte de colunas 100% NaN ou constantes
    cat_map = dict(zip(X.columns, cat_ind))
    cols_validas = [col for col in X.columns if X[col].nunique(dropna=True) > 1]
    X = X[cols_validas].copy()
    cat_cols = [i for i, col in enumerate(X.columns) if cat_map[col]]

    #Imputação pré-extração (mediana para numéricas, moda para categóricas)
    for i, col in enumerate(X.columns):
        if i in cat_cols:
            s = X[col].astype(object)
            moda = s.mode(dropna=True).iloc[0]
            X[col] = s.fillna(moda)
        else:
            s = pd.to_numeric(X[col], errors="coerce")
            X[col] = s.fillna(s.median())

    y_arr = pd.Series(y).astype("category").cat.codes.to_numpy()

    #Extração pymfe com one-hot-full
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfe = MFE(groups=GRUPOS_MFE)
        mfe.fit(
            X.to_numpy(dtype=object),
            y_arr,
            cat_cols=cat_cols,
            transform_cat="one-hot-full",
        )
        nomes, valores = mfe.extract()

    #Monta registro final (71 determinísticas + frac_missing = 72 metacaracterísticas)
    registro = {
        "tid": int(tid),
        "did": int(dataset.dataset_id),
        "name": dataset.name,
        "frac_missing": frac_missing,
    }
    for k, v in zip(nomes, valores):
        if k not in EXCLUIR_MFE:
            registro[k] = float(v) if v is not None else np.nan

    return registro


def auditar_resultado(df):
    cols_meta = [c for c in df.columns if c not in ("tid", "did", "name")]
    X_meta = df[cols_meta]

    nans = X_meta.isna().sum()
    nans = nans[nans > 0].sort_values(ascending=False)

    infs = np.isinf(X_meta.to_numpy(dtype=float)).sum()
    constantes = [c for c in cols_meta if X_meta[c].nunique(dropna=True) <= 1]

    log("--- AUDITORIA FINAL ---")
    log(f"Bases extraídas: {len(df)} | Metacaracterísticas: {len(cols_meta)}")
    log(f"Valores infinitos: {int(infs)}")
    log(f"Colunas constantes: {len(constantes)} {constantes if constantes else ''}")
    if len(nans) > 0:
        log(f"Colunas com NaN ({len(nans)}):")
        for col, qtd in nans.items():
            log(f"  {col}: {qtd}/{len(df)} ({qtd / len(df) * 100:.1f}%)")
    else:
        log("Nenhum NaN encontrado.")


def main():
    global ARQ_LOG
    ap = argparse.ArgumentParser(
        description="Extração das 72 metacaracterísticas (pymfe + frac_missing) das tasks do OpenML."
    )
    ap.add_argument("--dados", default="data", help="pasta de dados (padrão: data)")
    ap.add_argument("--task", type=int, help="extrai só esta task (útil para teste)")
    args = ap.parse_args()

    os.makedirs(args.dados, exist_ok=True)
    ARQ_LOG = os.path.join(args.dados, "extracao_metacaracteristicas.log")
    arq_saida = os.path.join(args.dados, "metacaracteristicas.csv")

    contagem = pd.read_csv(os.path.join(args.dados, "contagem_runs.csv"))
    if args.task:
        tids = [int(args.task)]
    else:
        tids = contagem["tid"].astype(int).tolist()

    if os.path.exists(arq_saida):
        df_salvo = pd.read_csv(arq_saida)
        tids_feitos = set(df_salvo["tid"].astype(int).tolist())
    else:
        df_salvo = pd.DataFrame()
        tids_feitos = set()

    log(f"início: {len(tids)} tasks na lista | {len(tids_feitos)} já salvas em {arq_saida}")

    ok, puladas, falhas = 0, 0, []
    t_inicio = time.time()

    try:
        for i, tid in enumerate(tids, start=1):
            if tid in tids_feitos:
                puladas += 1
                continue

            t0 = time.time()
            try:
                registro = extrair_task(tid)
            except Exception as e:
                log(f"[{i}/{len(tids)}] task {tid}: FALHOU ({e}). Rode de novo para tentar novamente.")
                falhas.append(tid)
                continue

            df_nova = pd.DataFrame([registro])
            df_salvo = pd.concat([df_salvo, df_nova], ignore_index=True)
            salvar_csv_atomico(df_salvo, arq_saida)
            tids_feitos.add(tid)
            ok += 1

            n_nan = int(df_nova.isna().sum().sum())
            log(
                f"[{i}/{len(tids)}] task {tid} ({registro['name']}): "
                f"ok em {time.time() - t0:.1f}s | NaNs={n_nan}"
            )
    except KeyboardInterrupt:
        log("interrompido pelo usuário. Rode de novo para continuar de onde parou.")
        return

    log(
        f"fim: {ok} extraídas, {puladas} já salvas, {len(falhas)} com falha "
        f"{falhas if falhas else ''} em {(time.time() - t_inicio) / 60:.1f} min"
    )

    if len(df_salvo) > 0:
        auditar_resultado(df_salvo)


if __name__ == "__main__":
    main()