import argparse
import os
import time
from datetime import datetime
import openml
import pandas as pd

FUNCAO = "area_under_roc_curve"
LOTE = 1000
MAX_OFFSET = 700_000
ESPERAS = [5, 15, 30, 60, 120]
COLUNAS_RUNS = ["run_id", "setup_id", "flow_id", "value", "upload_time"]

ARQ_LOG = None


def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    if ARQ_LOG:
        with open(ARQ_LOG, "a", encoding="utf-8") as f:
            f.write(linha + "\n")


def carregar_flows(pasta_dados):
    arq = os.path.join(pasta_dados, "flows.csv")
    if os.path.exists(arq):
        d = pd.read_csv(arq)
        return dict(zip(d.flow_id.astype(int), d.flow_name))
    return {}


def salvar_flows(flows, pasta_dados):
    arq = os.path.join(pasta_dados, "flows.csv")
    tmp = arq + ".tmp"
    d = pd.DataFrame({"flow_id": list(flows.keys()), "flow_name": list(flows.values())})
    d.sort_values("flow_id").to_csv(tmp, index=False)
    os.replace(tmp, arq)


def baixar_pagina(tid, offset):
    ultimo_erro = None
    for espera in [0] + ESPERAS:
        if espera:
            log(f"  nova tentativa em {espera}s (task {tid}, offset {offset}): {ultimo_erro}")
            time.sleep(espera)
        try:
            return openml.evaluations.list_evaluations(
                function=FUNCAO, tasks=[int(tid)], size=LOTE,
                offset=offset, output_format="dataframe")
        except openml.exceptions.OpenMLServerNoResult:
            return pd.DataFrame()
        except Exception as e:
            ultimo_erro = e
    raise ultimo_erro


def baixar_task(tid, esperado, pasta_dados, flows):
    pasta = os.path.join(pasta_dados, "runs", f"task_{tid}")
    os.makedirs(pasta, exist_ok=True)
    marca = os.path.join(pasta, "COMPLETA")
    if os.path.exists(marca):
        return None

    paginas = sorted(f for f in os.listdir(pasta)
                     if f.startswith("pagina_") and f.endswith(".csv.gz"))

    if paginas:
        ultima = pd.read_csv(os.path.join(pasta, paginas[-1]))
        if len(ultima) < LOTE:
            open(marca, "w").close()
            return (len(paginas) - 1) * LOTE + len(ultima)

    total = len(paginas) * LOTE
    offset = total
    if paginas:
        log(f"task {tid}: retomando do offset {offset} ({len(paginas)} páginas já gravadas)")

    t0 = time.time()
    terminou = False
    while offset < MAX_OFFSET:
        df = baixar_pagina(tid, offset)
        if df is None or len(df) == 0:
            terminou = True
            break

        faltando = [c for c in COLUNAS_RUNS + ["flow_name"] if c not in df.columns]
        if faltando:
            raise RuntimeError(f"colunas ausentes na resposta do OpenML: {faltando}. "
                               f"Colunas recebidas: {df.columns.tolist()}")

        novos = False
        for fid, nome in df[["flow_id", "flow_name"]].drop_duplicates().itertuples(index=False):
            if int(fid) not in flows:
                flows[int(fid)] = nome
                novos = True
        if novos:
            try:
                salvar_flows(flows, pasta_dados)
            except Exception as e:
                log(f"  aviso: não consegui gravar flows.csv agora ({e}); tento na próxima página")

        arq = os.path.join(pasta, f"pagina_{offset:07d}.csv.gz")
        tmp = arq + ".tmp"
        df[COLUNAS_RUNS].to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, arq)

        total += len(df)
        offset += LOTE

        if len(df) < LOTE:
            terminou = True
            break

        if (offset // LOTE) % 10 == 0:
            parte = f" ({total / esperado:.0%} do esperado)" if esperado else ""
            log(f"  task {tid}: {total} runs{parte}, {(time.time() - t0) / 60:.1f} min")
        time.sleep(0.2)

    if not terminou:
        raise RuntimeError(f"task {tid}: passou do limite de segurança de {MAX_OFFSET} runs")

    open(marca, "w").close()
    return total


def main():
    global ARQ_LOG
    ap = argparse.ArgumentParser(description="Extração de runs do OpenML (AUC)")
    ap.add_argument("--fase", choices=["1", "2", "todas"], default="1")
    ap.add_argument("--task", type=int, help="baixa só esta task (teste)")
    ap.add_argument("--dados", default="data", help="pasta de dados (padrão: dados)")
    ap.add_argument("--limite-fase1", type=int, default=70_000,
                    help="a fase 1 pega tasks com até este nº de runs (padrão: 70000)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.dados, "runs"), exist_ok=True)
    ARQ_LOG = os.path.join(args.dados, "extracao.log")

    contagem = pd.read_csv(os.path.join(args.dados, "contagem_runs.csv"))
    if args.task:
        alvo = contagem[contagem.tid == args.task]
        if alvo.empty:
            alvo = pd.DataFrame({"tid": [args.task], "runs_aprox": [None]})
    elif args.fase == "1":
        alvo = contagem[contagem.runs_aprox <= args.limite_fase1]
    elif args.fase == "2":
        alvo = contagem[contagem.runs_aprox > args.limite_fase1]
    else:
        alvo = contagem
    alvo = alvo.sort_values("runs_aprox")

    flows = carregar_flows(args.dados)
    log(f"início: {len(alvo)} tasks na lista (fase {args.task or args.fase})")

    ok, puladas, falhas = 0, 0, []
    t_inicio = time.time()
    try:
        for i, linha in enumerate(alvo.itertuples(index=False), start=1):
            tid = int(linha.tid)
            esperado = None if pd.isna(linha.runs_aprox) else int(linha.runs_aprox)
            t0 = time.time()
            try:
                total = baixar_task(tid, esperado, args.dados, flows)
            except Exception as e:
                log(f"[{i}/{len(alvo)}] task {tid}: FALHOU ({e}). Rode de novo para continuar.")
                falhas.append(tid)
                continue

            if total is None:
                puladas += 1
                log(f"[{i}/{len(alvo)}] task {tid}: já estava completa")
                continue

            ok += 1
            aviso = ""
            if esperado is not None and abs(total - esperado) > LOTE:
                aviso = f"  ATENÇÃO: esperado ~{esperado}"
            log(f"[{i}/{len(alvo)}] task {tid}: completa, {total} runs em "
                f"{(time.time() - t0) / 60:.1f} min{aviso}")
    except KeyboardInterrupt:
        log("interrompido pelo usuário. Rode de novo para continuar de onde parou.")
        return

    log(f"fim: {ok} baixadas, {puladas} já completas, {len(falhas)} com falha "
        f"{falhas if falhas else ''} em {(time.time() - t_inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()