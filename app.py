"""
Simulador numérico do modelo biomatemático:
Aguapés (A) -> Girinos (G) -> Sapos adultos (S) -> Escorpiões (E)

Nas duas páginas de simulação ("Simulações Numéricas" e "Aplicação em
Várzea das Flores"), o usuário informa, para cada espécie: a densidade
inicial A0/G0/S0/E0 (indivíduos/m²), a densidade máxima d_x (Tabela 2 do
artigo) e a área de habitat r_x — na página "Aplicação em Várzea das
Flores" a área já vem fixada com os valores reais da represa (Seção 4.1
do artigo) e não é editável, pois o usuário já está informando as áreas
adotadas. A capacidade suporte absoluta de cada espécie é sempre
k_x = d_x * r_x.

Internamente, em AMBAS as páginas, o sistema é sempre resolvido na forma
ADIMENSIONAL (Tabela 3 do artigo): a densidade inicial
é convertida para fração adimensional (A0/d_x) antes de integrar, e o
tempo é reescalado por n_a.

Depois de integrado, cada página apresenta os resultados em DOIS layouts
de 4 gráficos:
  1. Forma adimensional — cada população em relação à própria capacidade
     suporte (teto igual a 1), o que permite comparar visualmente a
     dinâmica das quatro espécies na mesma escala, mesmo que suas
     capacidades suporte reais difiram em várias ordens de grandeza.
  2. Forma dimensional — a mesma trajetória multiplicada por k_x = d_x *
     r_x, mostrando a população absoluta (número estimado de indivíduos)
     ao longo do tempo, em vez de apenas o valor final.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Modelo Biomatemático - Simulação", layout="wide")

# --------------------------------------------------------------------------
# 1. VALORES PADRÃO (Tabela 2 do enunciado) E METADADOS DOS PARÂMETROS
# --------------------------------------------------------------------------
DEFAULTS = {
    "n_a": 18.07,      # 1/ano
    "n_g": 6.6e3,      # 1/ano
    "n_e": 21.5,       # 1/ano
    "k_a": 14.0,       # aguapés/m²
    "k_g": 144.0,      # girinos/m²
    "k_s": 3.81e-4,    # sapos/m²
    "k_e": 0.17,       # escorpiões/m²
    "mu_a": 0.0,       # 1/ano (não fornecido na literatura -> padrão 0)
    "mu_s": 0.13,      # 1/ano
    "delta": 5.0e-3,   # 1/ano
    "alpha": 5.0,      # aguapés/m²
    "beta": 0.1,       # (?)
    "theta": 3.4e-2,   # escorpiões/m² (20% de k_e, valor atualizado da Tabela 2)
}

# --------------------------------------------------------------------------
# 1b. ESTUDO DE CASO — REPRESA DE VÁRZEA DAS FLORES
# --------------------------------------------------------------------------
# No artigo, os parâmetros k_a, k_g, k_s, k_e, alpha e theta representam
# valores absolutos, obtidos por k_x = d_x * r_x, sendo d_x a densidade de
# referência (Tabela 2) e r_x a área de habitat da espécie correspondente.
# Nesta página de aplicação, os valores DIMENSIONAIS de DEFAULTS (d_x) são
# usados diretamente na simulação — equivalente a adotar r_x = 1 m² para
# todas as espécies durante a integração numérica —, garantindo a
# consistência dimensional dos termos de acoplamento entre espécies (n_g*S
# na equação de G, delta*G na de S, beta*S na de E).
#
# As áreas reais abaixo (r_x) são usadas apenas DEPOIS da simulação, para
# traduzir a trajetória de densidade resultante (e não só o valor final)
# em uma estimativa de contagem absoluta de indivíduos na represa
# (k_x = d_x * r_x), sem realimentar o sistema de equações.
AREA_ESPELHO_AGUA = 3.76e6      # espelho d'água total (habitat do aguapé)

# Perímetro estimado pelo Índice de Desenvolvimento de Margem (Shoreline
# Development Index): D_s = P / (2*sqrt(pi*A))  =>  P = D_s * 2*sqrt(pi*A),
# adotando-se D_s = 4 (típico de represas dendríticas) => P ≈ 27.500 m.
D_S_ESTIMADO = 4.0
PERIMETRO_ESPELHO_AGUA = D_S_ESTIMADO * 2 * np.sqrt(np.pi * AREA_ESPELHO_AGUA)

# Girinos: em vez de uma fração da área do espelho d'água, considera-se uma
# faixa litorânea estreita ao longo do perímetro (largura entre 0,5 e 1 m,
# aqui 0,75 m como valor de referência), da qual apenas metade é
# efetivamente ocupada pelos girinos (distribuição não uniforme na margem).
LARGURA_MARGEM_GIRINO = 0.5     # m — largura da faixa litorânea considerada
FRACAO_OCUPACAO_GIRINO = 0.10   # fração da faixa efetivamente ocupada
AREA_GIRINO = FRACAO_OCUPACAO_GIRINO * PERIMETRO_ESPELHO_AGUA * LARGURA_MARGEM_GIRINO

LARGURA_APP = 30.0               # m — faixa de Área de Preservação Permanente
AREA_APP = PERIMETRO_ESPELHO_AGUA * LARGURA_APP   # faixa de APP, habitat de sapos e escorpiões
AREA_HABITAT_VZ = {"A": AREA_ESPELHO_AGUA, "G": AREA_GIRINO, "S": AREA_APP, "E": AREA_APP}

# --------------------------------------------------------------------------
# 1c. VALORES PADRÃO PARA A PÁGINA "SIMULAÇÕES NUMÉRICAS"
# --------------------------------------------------------------------------
# Nesta página, o usuário informa, para cada espécie, a densidade inicial
# (indivíduos/m², em blocos de destaque), a área r_x de habitat (também em
# destaque) e a densidade máxima d_x (num expander separado). A capacidade
# suporte absoluta é então k_x = d_x * r_x, usada para converter a densidade
# inicial em fração adimensional (A0/d_x) antes de resolver o sistema
# adimensional, e depois para converter a solução de volta em número
# absoluto de indivíduos (fração adimensional × k_x). Os valores padrão de
# área abaixo reaproveitam os mesmos números da aplicação em Várzea das
# Flores; os valores padrão de densidade inicial reproduzem as frações
# relativas que eram usadas antigamente como padrão (0,30 / 0,30 / 0,30 /
# 0,20) aplicadas sobre a densidade máxima de cada espécie.
AREAS_PADRAO_SIM = {"A": AREA_ESPELHO_AGUA, "G": AREA_GIRINO, "S": AREA_APP, "E": AREA_APP}
DENSIDADES_PADRAO_SIM = {"A": "k_a", "G": "k_g", "S": "k_s", "E": "k_e"}
FRACOES_PADRAO_SIM = {"A": 0.30, "G": 0.30, "S": 0.30, "E": 0.20}
DENSIDADE_INICIAL_PADRAO = {
    chave: FRACOES_PADRAO_SIM[chave] * DEFAULTS[DENSIDADES_PADRAO_SIM[chave]]
    for chave in ("A", "G", "S", "E")
}

# latex: símbolo usado nos rótulos (labels aceitam Markdown/LaTeX no Streamlit,
# o que resolve o problema dos "underlines" virando subscrito de verdade,
# e o texto fica no mesmo tamanho pequeno do rótulo do widget)
PARAM_INFO = {
    "n_a":   {"latex": r"n_a",     "unit": "1/ano",
              "desc": "Taxa de nascimento de aguapés."},
    "n_g":   {"latex": r"n_g",     "unit": "1/ano",
              "desc": "Taxa de nascimento dos girinos."},
    "n_e":   {"latex": r"n_e",     "unit": "1/ano",
              "desc": "Taxa de nascimento dos escorpiões."},
    "k_a":   {"latex": r"k_a",     "unit": "aguapés/m²",
              "desc": "Capacidade suporte de aguapés: quantidade máxima de plantas que o ambiente permite."},
    "k_g":   {"latex": r"k_g",     "unit": "girinos/m²",
              "desc": "Capacidade suporte de girinos: número máximo de girinos que o ecossistema sustenta, em um ambiente sem interferência de aguapés."},
    "k_s":   {"latex": r"k_s",     "unit": "sapos/m²",
              "desc": "Capacidade suporte de sapos adultos: quantidade máxima de sapos que o ambiente permite."},
    "k_e":   {"latex": r"k_e",     "unit": "escorpiões/m²",
              "desc": "Capacidade suporte de escorpiões: quantidade máxima que o ambiente permite."},
    "mu_a":  {"latex": r"\mu_a",   "unit": "1/ano",
              "desc": "Taxa de mortalidade de aguapés, que inclui morte natural e remoção de aguapés. (Sem valor de referência na literatura — padrão adotado: 0.)"},
    "mu_s":  {"latex": r"\mu_s",   "unit": "1/ano",
              "desc": "Taxa de mortalidade dos sapos adultos."},
    "delta": {"latex": r"\delta",  "unit": "1/ano",
              "desc": "Taxa de metamorfose, quando os girinos passam para a fase adulta (sapos)."},
    "alpha": {"latex": r"\alpha",  "unit": "aguapés/m²",
              "desc": "Taxa de saturação da função γ (regula o efeito da densidade de aguapés sobre a capacidade suporte dos girinos)."},
    "beta":  {"latex": r"\beta",   "unit": "adimensional",
              "desc": "Taxa de saturação da função λ (regula a predação de escorpiões pelos sapos)."},
    "theta": {"latex": r"\theta",  "unit": "escorpiões/m²",
              "desc": "Número mínimo de escorpiões no ambiente para que eles sejam foco predatório dos sapos."},
}

UNITS = {
    "A": "aguapés/m²",
    "G": "girinos/m²",
    "S": "sapos/m²",
    "E": "escorpiões/m²",
}
# Usadas na página "Simulações Numéricas": ali as populações são exibidas
# na forma adimensional (cada espécie em relação à sua própria capacidade
# suporte), sem unidade de densidade.
UNITS_REL = {
    "A": "",
    "G": "",
    "S": "",
    "E": "",
}
# Usadas quando o resultado é expresso em número absoluto de indivíduos
# (gráfico "dimensional" / população absoluta, nas duas páginas de
# simulação), depois de multiplicar a fração adimensional pela capacidade
# suporte absoluta k_x = d_x * r_x.
UNITS_ABS = {
    "A": "aguapés",
    "G": "girinos",
    "S": "sapos adultos",
    "E": "escorpiões",
}
NOMES = {
    "A": "Aguapés",
    "G": "Girinos",
    "S": "Sapos adultos",
    "E": "Escorpiões",
}
CORES = {"A": "#2e7d32", "G": "#6d4c41", "S": "#1565c0", "E": "#c62828"}

EPS = 1e-14  # evita divisões por zero

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def formata_cientifica(valor, casas=3):
    """
    Formata um número em notação científica no estilo "a×10^b" (com o
    expoente em sobrescrito unicode), em vez do formato "aeb" do Python,
    para manter a notação usada no restante do texto exibido ao usuário.
    """
    if valor == 0:
        return "0"
    sinal = "-" if valor < 0 else ""
    valor = abs(valor)
    expoente = int(np.floor(np.log10(valor)))
    mantissa = valor / 10 ** expoente
    mantissa_str = f"{mantissa:.{casas}f}"
    if float(mantissa_str) >= 10:
        mantissa /= 10
        expoente += 1
        mantissa_str = f"{mantissa:.{casas}f}"
    exp_str = str(expoente).translate(_SUPERSCRIPT)
    return f"{sinal}{mantissa_str}×10{exp_str}"


def rotulo(chave):
    """Monta o rótulo (em LaTeX, tamanho de label) para um number_input."""
    info = PARAM_INFO[chave]
    return f"${info['latex']}$ ({info['unit']})"

# --------------------------------------------------------------------------
# Modo de entrada da densidade/quantidade inicial: em densidade absoluta
# (indivíduos/m²) ou em % da densidade máxima d_x definida pelo usuário.
# Usado tanto na página "Simulações Numéricas" quanto na página
# "Aplicação em Várzea das Flores".
# --------------------------------------------------------------------------
MODO_DENSIDADE = "Densidade (indivíduos/m²)"
MODO_PERCENTUAL = "Porcentagem da densidade máxima ($d_x$)"

CENARIOS_INICIAIS = {
    "Início da invasão de aguapés": {
        # já na forma adimensional: valores diretamente em relação à
        # capacidade suporte de cada espécie
        "A0": 0.035,
        "G0": 0.07,
        "S0": 0.004,
        "E0": 0.02,
        "desc": (
            "Os aguapés começam a se estabelecer no ambiente, ainda em "
            "densidade baixa. A redução de luz e oxigênio já compromete "
            "levemente o desenvolvimento larval dos girinos, e a "
            "população de sapos adultos começa a sentir o efeito, com "
            "leve crescimento da população de escorpiões."
        ),
    },
    "Invasão severa": {
        "A0": 0.90,
        "G0": 0.05,
        "S0": 0.10,
        "E0": 0.80,
        "desc": (
            "Densidade de aguapés próxima da capacidade suporte do "
            "ambiente, comprometendo fortemente a fotossíntese e a "
            "oxigenação da água. O desenvolvimento dos girinos e a "
            "sobrevivência dos sapos adultos ficam severamente "
            "prejudicados, reduzindo o controle biológico e permitindo "
            "que a população de escorpiões-amarelos cresça — o cenário "
            "de maior risco de acidentes com humanos."
        ),
    },
}


def aplicar_cenario(nome):
    """Callback: preenche a densidade (ou porcentagem) inicial de cada
    espécie (A0, G0, S0, E0) no session_state a partir do cenário.

    O cenário é definido em frações relativas (forma adimensional). Se a
    página "Simulações Numéricas" estiver no modo MODO_PERCENTUAL, essas
    frações já são exatamente a porcentagem desejada (basta multiplicar por
    100). Se estiver no modo MODO_DENSIDADE, a fração é convertida em
    densidade (indivíduos/m²) multiplicando pela densidade máxima d_x
    atualmente definida pelo usuário (ou pelo padrão, se ele ainda não
    tiver mexido nesse campo) — a área não entra nessa conversão."""
    cenario = CENARIOS_INICIAIS[nome]
    modo = st.session_state.get("modo_inicial", MODO_DENSIDADE)
    if modo == MODO_PERCENTUAL:
        st.session_state["pct_A0"] = cenario["A0"] * 100
        st.session_state["pct_G0"] = cenario["G0"] * 100
        st.session_state["pct_S0"] = cenario["S0"] * 100
        st.session_state["pct_E0"] = cenario["E0"] * 100
    else:
        d_a = st.session_state.get("d_a", DEFAULTS["k_a"])
        d_g = st.session_state.get("d_g", DEFAULTS["k_g"])
        d_s = st.session_state.get("d_s", DEFAULTS["k_s"])
        d_e = st.session_state.get("d_e", DEFAULTS["k_e"])
        st.session_state["A0"] = cenario["A0"] * d_a
        st.session_state["G0"] = cenario["G0"] * d_g
        st.session_state["S0"] = cenario["S0"] * d_s
        st.session_state["E0"] = cenario["E0"] * d_e

# --------------------------------------------------------------------------
# 2. SISTEMA ADIMENSIONAL
# --------------------------------------------------------------------------
def sistema_adimensional(t, y, p):
    A, G, S, E = y

    dA = A * (1 - A) - p["mu_a"] * A

    dG = p["n_g"] * p["k"] * S * (1 - G * (1 + (A ** 2) / (p["alpha"] ** 2 + EPS))) - p["delta"] * G

    dS = (p["delta"] / (p["k"] + EPS)) * G * (1 - S) - p["mu_s"] * S

    dE = E * (
        p["n_e"] * (1 - E)
        - p["beta"] * S * (E ** 2 / (p["theta"] ** 2 + E ** 2 + EPS))
    )

    return [dA, dG, dS, dE]


def sistema_dimensional(t, y, dim):
    """
    Sistema dimensional (Equação \\eqref{modelo} do artigo, Seção 2), em
    unidades absolutas de indivíduos — sem normalizar pelas capacidades
    suporte. Usado apenas pela página "Simulações Numéricas": a simulação
    numérica é resolvida diretamente com a quantidade inicial absoluta
    informada pelo usuário; a conversão para a forma adimensional (fração da
    capacidade suporte) é feita só depois de integrado o sistema, apenas
    para exibir os gráficos numa escala comum e comparável entre as
    quatro espécies.

    `dim` deve conter as capacidades suporte absolutas (k_a, k_g, k_s,
    k_e = d_x * r_x) e os parâmetros alpha, theta também já convertidos
    para valores absolutos (alpha = d_alpha * r_a, theta = d_theta * r_e).
    """
    A, G, S, E = y
    k_a, k_g, k_s, k_e = dim["k_a"], dim["k_g"], dim["k_s"], dim["k_e"]
    n_a, n_g, n_e = dim["n_a"], dim["n_g"], dim["n_e"]
    mu_a, mu_s, delta = dim["mu_a"], dim["mu_s"], dim["delta"]
    alpha, beta, theta = dim["alpha"], dim["beta"], dim["theta"]

    dA = n_a * A * (1 - A / k_a) - mu_a * A

    gamma_A = k_g * (1 - (A ** 2) / (alpha ** 2 + A ** 2 + EPS))
    dG = n_g * S * (1 - G / (gamma_A + EPS)) - delta * G

    dS = delta * G * (1 - S / k_s) - mu_s * S

    lam_E = beta * S * E * (E ** 2 / (theta ** 2 + E ** 2 + EPS))
    dE = n_e * E * (1 - E / k_e) - lam_E

    return [dA, dG, dS, dE]


def adimensionaliza_parametros(dim):
    """Converte parâmetros dimensionais (Tabela 2) para adimensionais (Tabela 3)."""
    n_a = dim["n_a"]
    return {
        "n_g": dim["n_g"] / n_a,
        "n_e": dim["n_e"] / n_a,
        "k": dim["k_s"] / dim["k_g"],
        "mu_a": dim["mu_a"] / n_a,
        "mu_s": dim["mu_s"] / n_a,
        "delta": dim["delta"] / n_a,
        "alpha": dim["alpha"] / dim["k_a"],
        "beta": dim["beta"] * dim["k_s"] / n_a,
        "theta": dim["theta"] / dim["k_e"],
    }


# --------------------------------------------------------------------------
# 3. ANÁLISE DE TENDÊNCIA / ESTABILIZAÇÃO
# --------------------------------------------------------------------------
def analisa_populacao(t, y, tol=0.02):
    """
    Retorna: tendência (str), valor final, tempo de estabilização (ou None),
    índice de estabilização (ou None) e variação percentual.
    """
    y0, yf = y[0], y[-1]
    faixa = tol * max(abs(yf), 1e-12)

    # menor índice a partir do qual y permanece dentro da faixa até o fim
    idx_estavel = None
    for i in range(len(t)):
        if np.all(np.abs(y[i:] - yf) <= faixa):
            idx_estavel = i
            break
    t_estavel = t[idx_estavel] if idx_estavel is not None else None

    if abs(y0) < 1e-12:
        var_pct = np.inf if abs(yf) > 1e-12 else 0.0
    else:
        var_pct = 100.0 * (yf - y0) / abs(y0)

    if abs(yf - y0) <= tol * max(abs(y0), abs(yf), 1e-12):
        tendencia = "praticamente não se alterou"
    elif yf > y0:
        tendencia = "aumentou"
    else:
        tendencia = "diminuiu"

    return tendencia, yf, t_estavel, idx_estavel, var_pct


def gera_texto(resultados, t_max_anos, texto_convergencia=None, unidades=None):
    """
    unidades: dict opcional {"A":..., "G":..., "S":..., "E":...} com o texto
    de unidade a ser exibido após cada valor. Se None, usa UNITS (densidade
    real, usado na página "Aplicação em Várzea das Flores"). A página
    "Simulações Numéricas" passa UNITS_REL (strings vazias), já que ali as
    populações são exibidas na forma adimensional.
    """
    if unidades is None:
        unidades = UNITS

    linhas = []

    if texto_convergencia:
        linhas.append(texto_convergencia)
        linhas.append("")  # linha em branco antes da lista por espécie

    for chave in ["A", "G", "S", "E"]:
        nome = NOMES[chave]
        unidade = unidades[chave]
        tendencia, yf, t_estavel, _idx, var_pct = resultados[chave]

        sufixo = f" {unidade}" if unidade else ""
        if yf < 1e-9:
            desc_final = f"praticamente se extinguiu (valor final ≈ {formata_cientifica(yf)}{sufixo})"
        else:
            desc_final = f"se estabilizou em aproximadamente **{yf:.4g}{sufixo}**"

        if t_estavel is not None:
            desc_tempo = f", atingindo esse patamar por volta de **{t_estavel:.2f} ano(s)**"
        else:
            desc_tempo = f" (não atingiu estabilidade clara dentro do período simulado de {t_max_anos:.1f} ano(s))"

        if np.isfinite(var_pct):
            desc_var = f" — variação de aproximadamente {var_pct:+.1f}% em relação ao valor inicial"
        else:
            desc_var = ""

        linhas.append(
            f"- **{nome} ({chave})**: a população **{tendencia}** ao longo do tempo e {desc_final}{desc_tempo}{desc_var}."
        )
    return "\n".join(linhas)


def gera_texto_contagem(resultados, dim, areas, unidade_area="m²"):
    """
    Estima a contagem absoluta final de indivíduos de cada espécie, a partir
    da população relativa final (forma adimensional), da densidade de
    referência d_x (na página "Simulações Numéricas", os valores k_a, k_g,
    k_s, k_e informados em `dim` já representam densidades, pois adota-se
    r_x = 1 m² internamente — ver docstring no topo do arquivo) e da área de
    habitat r_x escolhida pelo usuário:

        contagem estimada = (população relativa final) × d_x × r_x
    """
    chave_para_kx = {"A": "k_a", "G": "k_g", "S": "k_s", "E": "k_e"}
    linhas = []
    for chave in ["A", "G", "S", "E"]:
        yf_relativo = resultados[chave][1]
        densidade_final = yf_relativo * dim[chave_para_kx[chave]]
        contagem = densidade_final * areas[chave]
        linhas.append(
            f"- **{NOMES[chave]} ({chave})**: aproximadamente "
            f"**{contagem:,.0f} indivíduos** "
            f"(densidade final ≈ {densidade_final:.4g} {UNITS[chave]} "
            f"× área de {areas[chave]:,.0f} {unidade_area})."
        )
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# 3b. PONTOS DE EQUILÍBRIO, JACOBIANO E ANÁLISE DE ESTABILIDADE
# --------------------------------------------------------------------------
NOMES_VAR = ("A", "G", "S", "E")


def resolve_E_positivo(S, p):
    """
    Resolve, para um dado S, as raízes reais e positivas de
        n_e(1-E)(theta^2+E^2) - beta*S*E^2 = 0
    (nulclina não trivial de E). Retorna lista ordenada de raízes.
    """
    n_e, beta, theta = p["n_e"], p["beta"], p["theta"]
    # n_e*E^3 - (n_e - beta*S)*E^2 + n_e*theta^2*E - n_e*theta^2 = 0
    coefs = [n_e, -(n_e - beta * S), n_e * theta ** 2, -n_e * theta ** 2]
    raizes = np.roots(coefs)
    reais_pos = sorted(r.real for r in raizes if abs(r.imag) < 1e-7 and r.real > 1e-9)
    return reais_pos


def calcula_equilibrios(p):
    """
    Calcula os pontos de equilíbrio P0..P7 do sistema adimensional,
    conforme a análise analítica do modelo. Retorna uma lista de dicts
    com nome, condição de existência (texto), se existe (bool) e coords
    (tupla (A,G,S,E) em variáveis adimensionais, ou None se não existe).
    """
    n_g, n_e, k = p["n_g"], p["n_e"], p["k"]
    mu_a, mu_s, delta = p["mu_a"], p["mu_s"], p["delta"]
    alpha, beta, theta = p["alpha"], p["beta"], p["theta"]

    pontos = []

    pontos.append(dict(nome="P0", condicao="sempre existe", existe=True,
                        coords=(0.0, 0.0, 0.0, 0.0)))

    pontos.append(dict(nome="P1", condicao="sempre existe", existe=True,
                        coords=(0.0, 0.0, 0.0, 1.0)))

    cond_ng = n_g > mu_s
    if cond_ng:
        G2 = k * (n_g - mu_s) / (delta + k * n_g + EPS)
        S2 = delta * (n_g - mu_s) / (n_g * (delta + k * mu_s) + EPS)
    else:
        G2 = S2 = None
    pontos.append(dict(nome="P2", condicao="n_g > mu_s", existe=cond_ng,
                        coords=(0.0, G2, S2, 0.0) if cond_ng else None))

    E1_raizes = resolve_E_positivo(S2, p) if cond_ng else []
    E1 = E1_raizes[0] if E1_raizes else None
    existe_p3 = cond_ng and E1 is not None
    pontos.append(dict(nome="P3", condicao="n_g > mu_s (com raiz E₁ > 0 da nulclina de E)",
                        existe=existe_p3,
                        coords=(0.0, G2, S2, E1) if existe_p3 else None))

    cond_mua = mu_a <= 1
    A4 = 1 - mu_a
    pontos.append(dict(nome="P4", condicao="mu_a ≤ 1", existe=cond_mua,
                        coords=(A4, 0.0, 0.0, 0.0) if cond_mua else None))

    pontos.append(dict(nome="P5", condicao="mu_a ≤ 1", existe=cond_mua,
                        coords=(A4, 0.0, 0.0, 1.0) if cond_mua else None))

    cond_p6 = cond_ng and cond_mua
    if cond_p6:
        den_G = alpha ** 2 * delta + k * n_g * (alpha ** 2 + A4 ** 2) + EPS
        G6 = alpha ** 2 * k * (n_g - mu_s) / den_G
        den_S = alpha ** 2 * delta + k * mu_s * (alpha ** 2 + A4 ** 2)
        S6 = alpha ** 2 * delta * (n_g - mu_s) / (n_g * den_S + EPS)
    else:
        G6 = S6 = None
    pontos.append(dict(nome="P6", condicao="n_g > mu_s e mu_a ≤ 1", existe=cond_p6,
                        coords=(A4, G6, S6, 0.0) if cond_p6 else None))

    E2_raizes = resolve_E_positivo(S6, p) if cond_p6 else []
    E2 = E2_raizes[0] if E2_raizes else None
    existe_p7 = cond_p6 and E2 is not None
    pontos.append(dict(nome="P7", condicao="n_g > mu_s e mu_a ≤ 1 (com raiz E₂ > 0 da nulclina de E)",
                        existe=existe_p7,
                        coords=(A4, G6, S6, E2) if existe_p7 else None))

    return pontos


def jacobiano(A, G, S, E, p):
    """Jacobiano do sistema adimensional, avaliado em (A,G,S,E)."""
    n_g, n_e, k = p["n_g"], p["n_e"], p["k"]
    mu_a, mu_s, delta = p["mu_a"], p["mu_s"], p["delta"]
    alpha, beta, theta = p["alpha"], p["beta"], p["theta"]
    a2 = alpha ** 2 + EPS
    den_E = E ** 2 + theta ** 2 + EPS

    J = np.zeros((4, 4))
    J[0, 0] = -2 * A - mu_a + 1

    J[1, 0] = -2 * A * G * S * k * n_g / a2
    J[1, 1] = S * k * n_g * (-(A ** 2) / a2 - 1) - delta
    J[1, 2] = k * n_g * (-G * (A ** 2 / a2 + 1) + 1)

    J[2, 1] = (delta / (k + EPS)) * (1 - S)
    J[2, 2] = -(G * delta) / (k + EPS) - mu_s

    J[3, 2] = -(E ** 3) * beta / den_E
    J[3, 3] = (2 * E ** 4 * S * beta) / (den_E ** 2) - (3 * E ** 2 * S * beta) / den_E + n_e * (1 - 2 * E)

    return J


def classifica_estabilidade(autovalores, tol=1e-8):
    """Classifica o ponto de equilíbrio a partir das partes reais dos autovalores."""
    partes_reais = np.real(autovalores)
    if np.any(np.abs(partes_reais) < tol):
        return "Caso degenerado (autovalor com parte real ≈ 0)"
    if np.all(partes_reais < 0):
        return "Estável"
    if np.all(partes_reais > 0):
        return "Instável (nó/fonte instável)"
    return "Instável (ponto de sela)"


def descreve_direcao(vetor, tol=0.05):
    """Descreve, em texto padrão, a direção de um autovetor real em termos de A, G, S, E."""
    norm = np.max(np.abs(vetor))
    if norm < 1e-12:
        return "direção nula"
    v = vetor / norm
    partes = []
    for nome, comp in zip(NOMES_VAR, v):
        if abs(comp) < tol:
            continue
        sinal = "aumento" if comp > 0 else "diminuição"
        partes.append(f"{sinal} de {nome}")
    if not partes:
        return "variação desprezível em todas as componentes"
    return ", ".join(partes)


def formata_autovetor(vetor):
    """Formata um autovetor (normalizado pela maior componente em módulo) como (vA, vG, vS, vE)."""
    norm = np.max(np.abs(vetor))
    if norm < 1e-12:
        v = vetor
    else:
        v = vetor / norm
    return "(" + ", ".join(f"{comp:.3g}" for comp in v) + ")"


def analisa_estabilidade_ponto(coords, p):
    """
    Para um ponto de equilíbrio (coords em variáveis adimensionais), calcula
    o jacobiano, os autovalores/autovetores e monta o texto padrão descrevendo
    as direções que APROXIMAM do ponto (autovalores com parte real negativa),
    identificando cada uma pelo nome lambda_i correspondente.
    Retorna: classificacao (str), texto_direcoes (str).
    """
    A, G, S, E = coords
    J = jacobiano(A, G, S, E, p)
    autovalores, autovetores = np.linalg.eig(J)
    classificacao = classifica_estabilidade(autovalores)

    linhas = []
    usados = set()
    for i, lam in enumerate(autovalores):
        if i in usados:
            continue
        if lam.real >= -1e-8:
            continue  # só reporta direções que aproximam (parte real < 0)

        if abs(lam.imag) < 1e-8:
            # autovalor real -> direção reta de aproximação
            v = np.real(autovetores[:, i])
            direcao = descreve_direcao(v)
            linhas.append(
                f"- Devido ao autovalor $\\lambda_{i+1}$, a trajetória se aproxima "
                f"desse ponto na direção **{direcao}**, correspondente ao autovetor "
                f"$v_{i+1} = {formata_autovetor(v)}$."
            )
        else:
            # par complexo conjugado -> aproximação em espiral no plano gerado
            # pela parte real e pela parte imaginária do autovetor
            j_conj = next((k for k, lam2 in enumerate(autovalores)
                            if k != i and k not in usados and abs(lam2 - np.conj(lam)) < 1e-6), None)
            if j_conj is not None:
                usados.add(j_conj)
                lam_label = f"$\\lambda_{i+1}$ e $\\lambda_{j_conj+1}$ (par conjugado)"
            else:
                lam_label = f"$\\lambda_{i+1}$"
            v = autovetores[:, i]
            direcao_re = descreve_direcao(np.real(v))
            direcao_im = descreve_direcao(np.imag(v))
            linhas.append(
                f"- Devido aos autovalores {lam_label}, a trajetória se aproxima "
                f"desse ponto em espiral, na direção **{direcao_re}** combinada com "
                f"**{direcao_im}**, correspondente ao autovetor "
                f"$v_{i+1} = {formata_autovetor(np.real(v))} + i\\,{formata_autovetor(np.imag(v))}$."
            )
        usados.add(i)

    if linhas:
        texto_direcoes = "\n".join(linhas)
    else:
        texto_direcoes = ("Nenhuma direção de aproximação (todos os autovalores têm "
                           "parte real ≥ 0 — o ponto não atrai trajetórias vizinhas).")

    return classificacao, texto_direcoes


def relatorio_estabilidade(dim, p, fatores=None):
    """
    Monta o relatório completo (texto Markdown) da verificação de estabilidade
    dos pontos de equilíbrio P0..P7, para os parâmetros atuais.

    fatores: tupla (f_A, f_G, f_S, f_E) usada para exibir as coordenadas.
    Se None, usa as capacidades suporte reais (dim["k_a"], ...) — caso da
    página "Aplicação em Várzea das Flores". A página "Simulações
    Numéricas" passa fatores=(1,1,1,1), pois ali as populações já são
    exibidas na forma adimensional.
    """
    if fatores is None:
        fatores = (dim["k_a"], dim["k_g"], dim["k_s"], dim["k_e"])

    pontos = calcula_equilibrios(p)
    blocos = []
    for pt in pontos:
        nome, existe, coords = pt["nome"], pt["existe"], pt["coords"]

        if not existe:
            blocos.append(
                f"### {nome}\n"
                f"**Condição de existência:** não satisfeita."
            )
            continue

        classificacao, texto_direcoes = analisa_estabilidade_ponto(coords, p)

        coords_dim = tuple(c * f for c, f in zip(coords, fatores))
        coords_txt = ", ".join(f"{v:.4g}" for v in coords_dim)
        nome_idx = nome[1:]  # número após o "P"

        blocos.append(
            f"### {nome} — {classificacao}\n"
            f"**Condição de existência:** satisfeita  \n"
            f"**Coordenadas:** $P_{{{nome_idx}}} = ({coords_txt})$  \n\n"
            f"**Direções que aproximam deste ponto:**\n{texto_direcoes}"
        )

    return "\n\n".join(blocos)


def identifica_convergencia(dim, p, dados, tol_proximo=0.02, tol_aproximando=0.20, fatores=None):
    """
    Compara o estado final da simulação (dados[...][-1]) com os pontos de
    equilíbrio existentes (calculados a partir dos parâmetros atuais) e
    identifica de qual ponto a trajetória mais se aproximou, para compor
    um texto padrão sobre "por qual caminho" as condições iniciais evoluíram.

    fatores: ver docstring de relatorio_estabilidade. Se None, usa as
    capacidades suporte reais (dim["k_a"], ...); a página "Simulações
    Numéricas" passa fatores=(1,1,1,1).
    """
    if fatores is None:
        fatores = (dim["k_a"], dim["k_g"], dim["k_s"], dim["k_e"])
    fatores = np.array(fatores)

    estado_final = np.array([dados[chave][-1] for chave in ["A", "G", "S", "E"]])

    pontos = calcula_equilibrios(p)
    melhor, melhor_dist = None, None
    for pt in pontos:
        if not pt["existe"]:
            continue
        coords_dim = np.array(pt["coords"]) * fatores
        # distância relativa às escalas de cada variável (capacidades suporte)
        dist_rel = np.sqrt(np.mean(((estado_final - coords_dim) / (fatores + EPS)) ** 2))
        if melhor is None or dist_rel < melhor_dist:
            melhor, melhor_dist = pt, dist_rel

    if melhor is None:
        return ("Não foi possível comparar o estado final da simulação com nenhum "
                "ponto de equilíbrio, pois nenhum deles existe para os parâmetros atuais.")

    classificacao, _ = analisa_estabilidade_ponto(melhor["coords"], p)
    nome = melhor["nome"]

    if melhor_dist <= tol_proximo:
        if classificacao == "Estável":
            return (f"Considerando as quatro populações em conjunto, as condições iniciais "
                     f"evoluíram seguindo a direção estável de **{nome}**, ponto de equilíbrio "
                     f"que o sistema efetivamente atingiu ao final do período simulado.")
        else:
            return (f"Considerando as quatro populações em conjunto, o estado final da "
                     f"simulação ficou muito próximo do ponto de equilíbrio **{nome}**, "
                     f"classificado como **{classificacao.lower()}**. Como esse ponto não é "
                     f"estável, essa proximidade tende a ser passageira — pequenas perturbações "
                     f"ou um período de simulação mais longo devem afastar o sistema dele.")
    elif melhor_dist <= tol_aproximando:
        return (f"Considerando as quatro populações em conjunto, as condições iniciais estão "
                 f"se aproximando da direção de **{nome}** (classificado como "
                 f"**{classificacao.lower()}**), mas o sistema ainda não atingiu completamente "
                 f"esse ponto dentro do período simulado — tente aumentar a duração da "
                 f"simulação para observar a estabilização completa.")
    else:
        return ("Considerando as quatro populações em conjunto, o estado final da simulação "
                 "ainda não se aproximou claramente de nenhum dos pontos de equilíbrio "
                 "calculados dentro do período simulado — tente aumentar a duração da "
                 "simulação.")


# --------------------------------------------------------------------------
# 4. SIDEBAR — NAVEGAÇÃO
# --------------------------------------------------------------------------
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Ir para:",
    ["Simulações Numéricas", "Modelo Matemático", "Como o Código Funciona",
     "Aplicação em Várzea das Flores"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# ==========================================================================
# PÁGINA 1 — MODELO MATEMÁTICO
# ==========================================================================
if pagina == "Modelo Matemático":
    st.title("Modelo Matemático")

    z1, z2 = st.columns(2)
    with z1:
        st.markdown("<h4 style='text-align:center'>Sistema dimensional</h4>", unsafe_allow_html=True)
        st.latex(r"""
        \begin{cases}
        \dfrac{dA}{dt} = n_aA\left(1-\dfrac{A}{k_a}\right) -\mu_aA\\[0.3cm]
        \dfrac{dG}{dt}=  n_{g} S\left(1-\dfrac{G}{\gamma(A)}\right) - \delta G\\[0.3cm]
        \dfrac{dS}{dt}=\delta G\left(1-\dfrac{S}{k_s}\right) - \mu_s S\\[0.3cm]
        \dfrac{dE}{dt}=n_eE\left(1-\dfrac{E}{k_e}\right)-\lambda(E)
        \end{cases}
        """)
        st.latex(r"\gamma(A) = k_g \left(1-\dfrac{A^2}{\alpha^2+A^2}\right) \qquad \lambda(E)=\beta SE\left(\dfrac{E^2}{\theta^2+E^2}\right)")
    with z2:
        st.markdown("<h4 style='text-align:center'>Sistema adimensional para cálculos</h4>", unsafe_allow_html=True)
        st.latex(r"""
        \begin{cases}
        \displaystyle \frac{d\bar{A}}{d\bar{t}} = \bar{A}\left(1-\bar{A}\right)-\bar{\mu_a}\bar{A}\\[0.4cm]
        \displaystyle \frac{d\bar{G}}{d\bar{t}} =\bar{n_g}\bar{k}\bar{S}\left(1-\bar{G}\left(1+\frac{\bar{A}^2}{\bar{\alpha}^2}\right)\right) - \bar{\delta}\bar{G}\\[0.4cm]
        \displaystyle \frac{d\bar{S}}{d\bar{t}} = \bar{\delta}\bar{k}^{-1}\bar{G}\left(1-\bar{S}\right) - \bar{\mu_s}\bar{S}\\[0.4cm]
        \displaystyle \frac{d{E}}{d{t}} = \bar{E}\left[\bar{n_e}\left(1-\bar{E}\right)-\bar{\beta}\bar{S}\left(\dfrac{\bar{E}^2}{\bar{\theta}^2+\bar{E}^2}\right)\right],
        \end{cases}
        """)

    st.markdown("---")
    st.subheader("Adimensionalização utilizada")
    st.latex(r"""
    \bar{A}=\frac{A}{k_a},\ \ \bar{G}=\frac{G}{k_g},\ \ \bar{S}=\frac{S}{k_s},\ \ \bar{E}=\frac{E}{k_e},\ \ \bar{t}=tn_a
    """)
    st.latex(r"""
    \bar{n}_g=\frac{n_g}{n_a},\ \ \bar{n}_e=\frac{n_e}{n_a},\ \ \bar{k}=\frac{k_s}{k_g},\ \ \bar{\mu}_a=\frac{\mu_a}{n_a},\ \ \bar{\mu}_s=\frac{\mu_s}{n_a},\ \ \bar{\alpha}=\frac{\alpha}{k_a},\ \ \bar{\beta}=\frac{\beta k_s}{n_a},\ \ \bar{\delta}=\frac{\delta}{n_a},\ \ \bar{\theta}=\frac{\theta}{k_e}
    """)

    st.markdown("---")
    st.subheader("Significado dos parâmetros")
    for chave, info in PARAM_INFO.items():
        col_simb, col_desc = st.columns([1, 6])
        with col_simb:
            st.latex(info["latex"])
        with col_desc:
            st.markdown(f"**Unidade:** {info['unit']}  \n{info['desc']}")

    st.markdown("---")
    st.subheader("Valores padrão adotados")
    st.caption(
        "Os valores abaixo são densidades de referência $d_x$ (Tabela 2 do "
        "artigo), não as capacidades suporte absolutas $k_x$: "
        "$k_x = d_x \\cdot r_x$ só é calculado depois, ao multiplicar pela "
        "área de habitat $r_x$ de cada espécie (fixa na página de "
        "aplicação, ou definida pelo usuário na de simulações)."
    )

    # Nesta tabela, os valores exibidos para k_a/k_g/k_s/k_e/alpha/theta são
    # densidades de referência (d_x), não as capacidades suporte absolutas
    # (k_x = d_x * r_x). Por isso, aqui — e só aqui — o símbolo mostrado é
    # trocado para d_x, mantendo o restante do app (rótulos de widgets,
    # significado dos parâmetros etc.) como está.
    SIMBOLOS_VALORES_PADRAO = {
        "k_a": r"d_a", "k_g": r"d_g", "k_s": r"d_s", "k_e": r"d_e",
        "alpha": r"d_\alpha", "theta": r"d_\theta",
    }

    col_p, col_v, col_u = st.columns([1, 2, 3])
    with col_p:
        st.markdown("**Parâmetro**")
    with col_v:
        st.markdown("**Valor**")
    with col_u:
        st.markdown("**Unidade**")

    for chave in DEFAULTS:
        info = PARAM_INFO[chave]
        simbolo = SIMBOLOS_VALORES_PADRAO.get(chave, info["latex"])
        col_p, col_v, col_u = st.columns([1, 2, 3])
        with col_p:
            st.latex(simbolo)
        with col_v:
            st.markdown(f"{DEFAULTS[chave]:.4g}")
        with col_u:
            st.markdown(info["unit"])
    st.caption(
        r"$\mu_a$ adotado por padrão é 0, mas pode "
        "ser alterado livremente na página de simulações, a fim de "
        "verificar como a retirada mecânica de aguapés afeta "
        "a dinâmica populacional."
    )

    
    # st.markdown("---")
    # st.subheader("Convenção adotada nas Simulações Numéricas")
    # st.markdown(
        # "Na página **Simulações Numéricas**, os resultados são sempre "
        # "mostrados na **forma adimensional**. Assim, "
        # "cada população é sempre mostrada em relação à sua própria "
        # "capacidade suporte, o que permite comparar visualmente a dinâmica "
        # "das quatro espécies na mesma escala, mesmo que suas capacidades "
        # "suporte reais (tabela acima) sejam muito diferentes entre si. Os "
        # "valores reais de $k_a, k_g, k_s, k_e$ continuam sendo usados "
        # "internamente para calcular corretamente os parâmetros "
        # "adimensionais do sistema ($\\bar{k}, \\bar{\\alpha}, \\bar{\\beta}, "
        # "\\bar{\\theta}$). A aplicação do modelo com valores absolutos de "
        # "população, obtidos a partir de áreas de habitat reais, é tratada "
        # "separadamente na página **Aplicação em Várzea das Flores**."
    #)

# ==========================================================================
# PÁGINA 2 — COMO O CÓDIGO FUNCIONA
# ==========================================================================
elif pagina == "Como o Código Funciona":
    st.title("Como o Código Funciona")

    st.markdown(
        "Esta página explica, passo a passo e na mesma ordem em que o código "
        "executa, o que acontece depois que você clica em **Rodar simulação**. "
        "As fórmulas usadas (pontos de equilíbrio, condições de existência, "
        "Jacobiano etc.) foram deduzidas previamente a partir do modelo "
        "matemático, o código apenas as implementa e resolve numericamente."
    )

    st.markdown("---")
    st.subheader("Parte 1 — Verificação de estabilidade dos pontos de equilíbrio")
    st.markdown(
        "Essa verificação roda **antes** da simulação numérica propriamente "
        "dita, usando apenas os parâmetros informados (não depende das "
        "condições iniciais). Ordem de execução:"
    )
    st.markdown(
        """
1. **Adimensionaliza os parâmetros.** Os parâmetros dimensionais informados
   nos campos são convertidos para as variáveis adimensionais (as mesmas
   relações mostradas na página **Modelo Matemático**).

2. **Verifica a condição de existência de cada ponto de equilíbrio (P0 a P7).**
   Para cada um dos 8 pontos, o código testa a desigualdade/condição
   correspondente (por exemplo, $n_g > \\mu_s$ ou $\\mu_a \\le 1$) e, se ela for
   satisfeita, calcula as coordenadas do ponto pelas fórmulas já deduzidas
   analiticamente. Se a condição não é satisfeita, o ponto é reportado como
   "não existe" e nenhuma outra conta é feita para ele.

3. **Encontra a coordenada de equilíbrio de E nos pontos P3 e P7.**
   Esses dois pontos dependem da raiz positiva de um polinômio de grau 3 em
   $E$ (a nulclina não trivial de $E$). O código monta os coeficientes desse
   polinômio e chama a função nativa **`numpy.roots`** para obter todas as
   raízes (reais e complexas); em seguida filtra apenas as raízes reais e
   positivas e usa a menor delas.

4. **Monta a matriz Jacobiana do sistema adimensional**, avaliada nas
   coordenadas de cada ponto de equilíbrio que existe. As expressões de cada
   entrada da Jacobiana (as derivadas parciais) também já haviam sido
   calculadas previamente; o código só substitui os valores numéricos.

5. **Calcula autovalores e autovetores da Jacobiana** chamando a função
   nativa **`numpy.linalg.eig`** sobre a matriz montada no passo anterior.

6. **Classifica a estabilidade do ponto** observando o sinal da parte real
   dos autovalores (obtida com **`numpy.real`**):
   - todas as partes reais negativas → **estável**;
   - todas positivas → **instável (nó/fonte instável)**;
   - sinais mistos → **instável (ponto de sela)**;
   - alguma parte real ≈ 0 → **caso degenerado**.

7. **Descreve a direção de aproximação ao ponto.** Para cada autovalor com
   parte real negativa (ou seja, cada direção que atrai trajetórias vizinhas),
   o código lê o autovetor correspondente e descreve, em palavras, quais
   variáveis ($A$, $G$, $S$ ou $E$) aumentam ou diminuem nessa direção. Quando
   dois autovalores formam um par complexo conjugado, a aproximação é em
   espiral: o código usa a parte real (**`numpy.real`**) e a parte imaginária
   (**`numpy.imag`**) do autovetor para descrever as duas direções combinadas.

8. **Exibe as coordenadas do ponto de equilíbrio.** Na página
   **Simulações Numéricas**, isso é feito diretamente na forma
   adimensional (sem reconversão), a mesma usada nos gráficos; na página
   **Aplicação em Várzea das Flores**, as coordenadas são multiplicadas
   pela respectiva capacidade suporte real antes de aparecerem no texto.
   Em ambos os casos, o resultado é o texto exibido dentro do expansor
   **"Ver verificação de estabilidade"**.
        """
    )

    st.markdown("---")
    st.subheader("Parte 2 — Simulação numérica e gráficos")
    st.markdown(
        "Depois da verificação de estabilidade, e já com as condições "
        "iniciais que você informou, o código roda a simulação propriamente "
        "dita, nesta ordem:"
    )
    st.markdown(
        """
1. **Usa diretamente as condições iniciais informadas** ($A_0, G_0, S_0,
   E_0$). Na página **Simulações Numéricas**, elas já são fornecidas na
   forma adimensional (ou seja, como a própria variável adimensional); na
   página **Aplicação em Várzea das Flores**, são fornecidas como
   densidade e o código as divide pela respectiva capacidade suporte real
   para obter a condição inicial adimensional.

2. **Define o tempo adimensional de simulação**, multiplicando a duração
   escolhida (em anos) por $n_a$, e cria o vetor de instantes em que a
   solução será avaliada com a função nativa **`numpy.linspace`** (3000
   pontos igualmente espaçados).

3. **Resolve numericamente o sistema de equações diferenciais** (o sistema
   adimensional mostrado na página Modelo Matemático) chamando a função
   nativa **`scipy.integrate.solve_ivp`**, com o método `"LSODA"`
   (adequado para sistemas que podem ficar rígidos/"stiff") e tolerâncias de
   erro relativa e absoluta ajustadas para maior precisão.

4. **Converte apenas o eixo do tempo de volta para anos** (dividindo por
   $n_a$). Na página **Simulações Numéricas**, as quatro séries temporais
   das populações permanecem na forma adimensional, ficando todas na mesma
   escala (entre 0 e aproximadamente 1); na página **Aplicação em Várzea
   das Flores**, cada série é multiplicada pela respectiva capacidade
   suporte real para retornar à densidade dimensional.

5. **Verifica, para cada população, se e quando ela se estabiliza**, ou seja,
   a partir de que instante os valores passam a permanecer dentro de uma
   faixa de 2% em torno do valor final observado no período simulado (essa
   lógica de comparação foi definida por mim, não é uma função pronta de
   biblioteca).

6. **Monta a figura com os 4 gráficos** (grade 2×2, um por população) usando
   as funções nativas do **`matplotlib.pyplot`**: `subplots` para criar a
   grade de eixos, `ax.plot` para desenhar cada curva, `ax.scatter` e
   `ax.axvline` para marcar o ponto e a linha vertical de estabilização, e
   `ax.set_title` / `ax.set_xlabel` / `ax.set_ylabel` / `ax.grid` para os
   textos e a grade de cada gráfico.

7. **Exibe a figura no aplicativo** com `st.pyplot`.

8. **Compara o estado final da simulação com os pontos de equilíbrio**
   calculados na Parte 1 (recalculados para os mesmos parâmetros atuais) e
   identifica de qual ponto o sistema mais se aproximou, com base na
   distância entre o estado final e as coordenadas de cada ponto existente.
   Com isso, monta o texto de interpretação exibido em **"Interpretação dos
   resultados"**, logo abaixo dos gráficos.
        """
    )

# ==========================================================================
# PÁGINA 4 — APLICAÇÃO EM VÁRZEA DAS FLORES
# ==========================================================================
elif pagina == "Aplicação em Várzea das Flores":
    st.title("Aplicação em Várzea das Flores")

    st.markdown(
        "A Represa de Várzea das Flores (Vargem das Flores) fica entre Contagem "
        "e Betim, na Região Metropolitana de Belo Horizonte, com um espelho "
        "d'água de **3,76 km²**. Toda a "
        "bacia está protegida pela APA Vargem das Flores (12,2 mil hectares, "
        "criada em 2006)."
    )
    st.markdown(
        "A urbanização de Contagem e Betim gera esgoto deficitário, descarte "
        "irregular de lixo e efluentes industriais/de mineração lançados nos "
        "cursos d'água, o que degrada a qualidade da água, favorece a "
        "eutrofização e gera a condição ideal para a proliferação do aguapé. Isso "
        "compromete a reprodução do sapo-cururu e, por consequência, afeta a "
        "densidade de escorpiões-amarelos na região. Entre 2021 e 2026, o "
        "Sinan registrou 12.536 acidentes escorpiônicos na Região Metropolitana de Belo Horizonte (168 graves, "
        "com maior gravidade em crianças), reforçando a importância de "
        "entender essa dinâmica populacional para orientar medidas preventivas."
    )

    st.subheader("Áreas de habitat na represa")
    st.markdown(
        "Internamente, o modelo usa os valores padrão adotados para os parâmetros de região de habitat para cada população:"
    )
    st.markdown(
        "- **Aguapés**: flutuam livremente sobre toda a superfície. Sua área de ocupação corresponde ao "
        "espelho d'água, **3,76×10⁶ m²**.\n"
        "- **Girinos**: concentram-se nas margens rasas por fatores como temperatura, "
        "oviposição, abrigo e alimento. Consideramos sua ocupação restrita a uma "
        "**faixa litorânea de 0,75 m de largura** ao longo do perímetro do espelho "
        "d'água, da qual apenas **50%** é efetivamente ocupada (distribuição não "
        "uniforme na margem), resultando em **≈1,03×10⁴ m²**.\n"
        "- **Sapos adultos e escorpiões-amarelos**: restritos à faixa de "
        "Área de Preservação Permanente (30 m ao redor do reservatório, "
        "aproximadamente **8,25×10⁵ m²**) pela preferência dos sapos de se manterem "
        "em locais úmidos e considerando que predador e presa compartilhando o mesmo habitat. "
    )
    st.caption(
        "Como o perímetro real do espelho d'água não é conhecido, ele foi "
        "estimado pelo Índice de Desenvolvimento de Margem "
        r"($D_s = P / (2\sqrt{\pi A})$), adotando-se $D_s = 4$ (típico de "
        "represas dendríticas), o que resulta em um perímetro estimado de "
        "≈27.500 m. Esse perímetro é usado tanto para a faixa de APP de "
        "30 m (≈825.000 m², sapos e escorpiões) quanto para a faixa "
        "litorânea de 0,75 m com 50% de ocupação (≈1,03×10⁴ m², girinos)."
    )

    st.markdown("---")
    st.subheader("Simulação — Várzea das Flores")
    st.markdown(
        "Os campos abaixo já vêm preenchidos com os valores da Tabela 2."
        " Ajuste se desejar e clique em **Rodar simulação**."
    )

    st.markdown("**Densidades iniciais**")
    modo_inicial_vz = st.radio(
        "Como deseja informar a condição inicial?",
        [MODO_DENSIDADE, MODO_PERCENTUAL],
        horizontal=True, key="modo_inicial_vz",
    )

    # densidade máxima (d_a_vz etc.) atualmente definida pelo usuário no
    # expander "Parâmetros do modelo" abaixo — lida via session_state
    # porque, na ordem do script, esse expander só é criado mais adiante;
    # o valor aqui já reflete a última interação do usuário (ou o padrão
    # da Tabela 2, se ele ainda não tiver mexido no campo).
    d_a_vz_atual = st.session_state.get("d_a_vz", DEFAULTS["k_a"])
    d_g_vz_atual = st.session_state.get("d_g_vz", DEFAULTS["k_g"])
    d_s_vz_atual = st.session_state.get("d_s_vz", DEFAULTS["k_s"])
    d_e_vz_atual = st.session_state.get("d_e_vz", DEFAULTS["k_e"])

    if modo_inicial_vz == MODO_DENSIDADE:
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            A0_vz = st.number_input("$A_0$: Aguapés (aguapés/m²)", min_value=0.0,
                                     value=st.session_state.get("A0_vz", 0.30 * DEFAULTS["k_a"]),
                                     format="%.4g", key="A0_vz")
            st.caption(f"≈ {A0_vz / d_a_vz_atual * 100:.4g}% de $d_a$")
        with v2:
            G0_vz = st.number_input("$G_0$: Girinos (girinos/m²)", min_value=0.0,
                                     value=st.session_state.get("G0_vz", 0.35 * DEFAULTS["k_g"]),
                                     format="%.4g", key="G0_vz")
            st.caption(f"≈ {G0_vz / d_g_vz_atual * 100:.4g}% de $d_g$")
        with v3:
            S0_vz = st.number_input("$S_0$: Sapos (sapos/m²)", min_value=0.0,
                                     value=st.session_state.get("S0_vz", 0.40 * DEFAULTS["k_s"]),
                                     format="%.4g", key="S0_vz")
            st.caption(f"≈ {S0_vz / d_s_vz_atual * 100:.4g}% de $d_s$")
        with v4:
            E0_vz = st.number_input("$E_0$: Escorpiões (escorpiões/m²)", min_value=0.0,
                                     value=st.session_state.get("E0_vz", 0.20 * DEFAULTS["k_e"]),
                                     format="%.4g", key="E0_vz")
            st.caption(f"≈ {E0_vz / d_e_vz_atual * 100:.4g}% de $d_e$")
    else:
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            pct_A0_vz = st.number_input("$A_0$: Aguapés (% de $d_a$)",
                                         min_value=0.0, max_value=100.0,
                                         value=st.session_state.get("pct_A0_vz", 30.0),
                                         format="%.4g", key="pct_A0_vz")
            A0_vz = pct_A0_vz / 100 * d_a_vz_atual
            st.caption(f"≈ {A0_vz:.4g} aguapés/m²")
        with v2:
            pct_G0_vz = st.number_input("$G_0$: Girinos (% de $d_g$)",
                                         min_value=0.0, max_value=100.0,
                                         value=st.session_state.get("pct_G0_vz", 35.0),
                                         format="%.4g", key="pct_G0_vz")
            G0_vz = pct_G0_vz / 100 * d_g_vz_atual
            st.caption(f"≈ {G0_vz:.4g} girinos/m²")
        with v3:
            pct_S0_vz = st.number_input("$S_0$: Sapos (% de $d_s$)",
                                         min_value=0.0, max_value=100.0,
                                         value=st.session_state.get("pct_S0_vz", 40.0),
                                         format="%.4g", key="pct_S0_vz")
            S0_vz = pct_S0_vz / 100 * d_s_vz_atual
            st.caption(f"≈ {S0_vz:.4g} sapos/m²")
        with v4:
            pct_E0_vz = st.number_input("$E_0$: Escorpiões (% de $d_e$)",
                                         min_value=0.0, max_value=100.0,
                                         value=st.session_state.get("pct_E0_vz", 20.0),
                                         format="%.4g", key="pct_E0_vz")
            E0_vz = pct_E0_vz / 100 * d_e_vz_atual
            st.caption(f"≈ {E0_vz:.4g} escorpiões/m²")

    with st.expander("Parâmetros do modelo (valores da Tabela 2)"):
        st.caption(
            "$d_a, d_g, d_s, d_e$ são as densidades máximas de referência "
            "da Tabela 2 do artigo (indivíduos/m²); junto com a área real "
            "de habitat na represa (fixada acima), determinam a "
            "capacidade suporte $k_x = d_x \\times r_x$ de cada espécie — "
            "não é o $k_x$ que deve ser informado aqui."
        )
        q1, q2, q3 = st.columns(3)
        with q1:
            n_a_vz = st.number_input(rotulo("n_a"), value=DEFAULTS["n_a"], format="%.6f", key="n_a_vz")
            n_g_vz = st.number_input(rotulo("n_g"), value=DEFAULTS["n_g"], format="%.6f", key="n_g_vz")
            n_e_vz = st.number_input(rotulo("n_e"), value=DEFAULTS["n_e"], format="%.6f", key="n_e_vz")
            d_a_vz = st.number_input("$d_a$ — Aguapés (aguapés/m²)", min_value=1e-12,
                                      value=DEFAULTS["k_a"], format="%.6g", key="d_a_vz")
        with q2:
            d_g_vz = st.number_input("$d_g$ — Girinos (girinos/m²)", min_value=1e-12,
                                      value=DEFAULTS["k_g"], format="%.6g", key="d_g_vz")
            d_s_vz = st.number_input("$d_s$ — Sapos adultos (sapos/m²)", min_value=1e-12,
                                      value=DEFAULTS["k_s"], format="%.6g", key="d_s_vz")
            d_e_vz = st.number_input("$d_e$ — Escorpiões (escorpiões/m²)", min_value=1e-12,
                                      value=DEFAULTS["k_e"], format="%.6g", key="d_e_vz")
            mu_a_vz = st.number_input(rotulo("mu_a"), value=DEFAULTS["mu_a"], format="%.6f", key="mu_a_vz")
        with q3:
            mu_s_vz = st.number_input(rotulo("mu_s"), value=DEFAULTS["mu_s"], format="%.6f", key="mu_s_vz")
            delta_vz = st.number_input(rotulo("delta"), value=DEFAULTS["delta"], format="%.6e", key="delta_vz")
            alpha_vz = st.number_input(rotulo("alpha"), min_value=1e-9, value=DEFAULTS["alpha"], format="%.6g", key="alpha_vz")
            beta_vz = st.number_input(rotulo("beta"), value=DEFAULTS["beta"], format="%.6f", key="beta_vz")
        theta_vz = st.number_input(rotulo("theta"), value=DEFAULTS["theta"], format="%.6g", key="theta_vz")

    st.subheader("Tempo de simulação")
    t_max_vz = st.slider("Duração máxima da simulação (ano(s))", min_value=1, max_value=100,
                          value=20, key="t_max_vz")

    dim_vz = dict(n_a=n_a_vz, n_g=n_g_vz, n_e=n_e_vz, k_a=d_a_vz, k_g=d_g_vz, k_s=d_s_vz,
                  k_e=d_e_vz, mu_a=mu_a_vz, mu_s=mu_s_vz, delta=delta_vz, alpha=alpha_vz,
                  beta=beta_vz, theta=theta_vz)

    rodar_vz = st.button("Rodar simulação", type="primary", key="rodar_vz")

    if rodar_vz:
        if n_a_vz <= 0:
            st.error("n_a deve ser positivo (é usado para adimensionalizar o tempo).")
            st.stop()
        if d_a_vz <= 0 or d_g_vz <= 0 or d_s_vz <= 0 or d_e_vz <= 0:
            st.error("As densidades máximas (d_a, d_g, d_s, d_e) devem ser positivas.")
            st.stop()

        p_vz = adimensionaliza_parametros(dim_vz)

        y0_bar_vz = [A0_vz / d_a_vz, G0_vz / d_g_vz, S0_vz / d_s_vz, E0_vz / d_e_vz]

        t_bar_max_vz = t_max_vz * n_a_vz
        t_bar_eval_vz = np.linspace(0, t_bar_max_vz, 3000)

        sol_vz = solve_ivp(
            sistema_adimensional, [0, t_bar_max_vz], y0_bar_vz,
            args=(p_vz,), t_eval=t_bar_eval_vz, method="LSODA",
            rtol=1e-8, atol=1e-10,
        )

        if not sol_vz.success:
            st.error(f"A integração numérica falhou: {sol_vz.message}")
            st.stop()

        t_dim_vz = sol_vz.t / n_a_vz
        ordem_vz = ["A", "G", "S", "E"]

        # -- forma adimensional: cada espécie em relação à própria
        #    capacidade suporte (teto em 1). sol_vz.y já está
        #    nessa forma, pois o sistema foi integrado adimensionalmente.
        dados_vz_rel = {"A": sol_vz.y[0], "G": sol_vz.y[1], "S": sol_vz.y[2], "E": sol_vz.y[3]}

        # -- forma dimensional (densidade real, indivíduos/m²): fração
        #    adimensional × densidade máxima d_x.
        k_dens_vz = {"A": d_a_vz, "G": d_g_vz, "S": d_s_vz, "E": d_e_vz}
        dados_vz_dens = {chave: dados_vz_rel[chave] * k_dens_vz[chave] for chave in ordem_vz}

        # -- população absoluta (nº de indivíduos): densidade × área real
        #    de habitat na represa (AREA_HABITAT_VZ), aplicada à trajetória
        #    inteira (não só ao valor final).
        dados_vz_abs = {chave: dados_vz_dens[chave] * AREA_HABITAT_VZ[chave] for chave in ordem_vz}

        # fatores para exibir pontos de equilíbrio e comparar a convergência
        # nas mesmas unidades absolutas dos gráficos/textos abaixo
        fatores_abs_vz = tuple(k_dens_vz[chave] * AREA_HABITAT_VZ[chave] for chave in ordem_vz)

        # o tempo de estabilização é o mesmo em qualquer uma das três formas
        # (adimensional, densidade ou absoluta), pois escalar por uma
        # constante positiva não muda o critério relativo de estabilização
        resultados_vz = {chave: analisa_populacao(t_dim_vz, dados_vz_abs[chave]) for chave in ordem_vz}

        st.subheader("Resultados — forma adimensional")
        st.caption(
            "Para permitir a comparação visual entre as quatro espécies "
            "com capacidades suporte diferentes, cada gráfico abaixo "
            "expressa a população em relação à própria capacidade suporte "
            "da espécie, variando de 0 a 1."
        )
        fig_vz_rel, axs_vz_rel = plt.subplots(2, 2, figsize=(12, 8))
        for ax, chave in zip(axs_vz_rel.flat, ordem_vz):
            _tend, _yf, t_estavel, idx_estavel, _var = resultados_vz[chave]

            ax.plot(t_dim_vz, dados_vz_rel[chave], color=CORES[chave], linewidth=2)

            if idx_estavel is not None:
                t_marco = t_dim_vz[idx_estavel]
                y_marco = dados_vz_rel[chave][idx_estavel]
                ax.scatter([t_marco], [y_marco], color=CORES[chave], zorder=5, s=40)
                ax.axvline(t_marco, color=CORES[chave], linestyle="--", linewidth=1, alpha=0.4)
                ax.set_title(f"{NOMES[chave]} ({chave}) — estabilizou em {t_marco:.1f} ano(s)")
            else:
                ax.set_title(f"{NOMES[chave]} ({chave})")

            ax.set_xlabel("Tempo (ano(s))")
            ax.set_ylabel("População (forma adimensional)")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        fig_vz_rel.tight_layout()
        st.pyplot(fig_vz_rel)

        st.subheader("Resultados — forma dimensional")
        st.caption(
            "Mesma trajetória acima, convertida em número absoluto de "
            "indivíduos. Cada gráfico numa escala própria, já que as populações reais "
            "diferem em densidade."
        )
        fig_vz_abs, axs_vz_abs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, chave in zip(axs_vz_abs.flat, ordem_vz):
            _tend, yf, t_estavel, idx_estavel, _var = resultados_vz[chave]

            ax.plot(t_dim_vz, dados_vz_abs[chave], color=CORES[chave], linewidth=2)

            if idx_estavel is not None:
                t_marco = t_dim_vz[idx_estavel]
                y_marco = dados_vz_abs[chave][idx_estavel]
                ax.scatter([t_marco], [y_marco], color=CORES[chave], zorder=5, s=40)
                ax.axvline(t_marco, color=CORES[chave], linestyle="--", linewidth=1, alpha=0.4)
                ax.set_title(f"{NOMES[chave]} ({chave}) — estabilizou em {t_marco:.1f} ano(s)")
            else:
                ax.set_title(f"{NOMES[chave]} ({chave})")

            ax.set_xlabel("Tempo (ano(s))")
            ax.set_ylabel("Indivíduos")
            ax.grid(alpha=0.3)
        fig_vz_abs.tight_layout()
        st.pyplot(fig_vz_abs)

        st.subheader("Interpretação dos resultados")
        texto_convergencia_vz = identifica_convergencia(dim_vz, p_vz, dados_vz_abs, fatores=fatores_abs_vz)
        st.markdown(gera_texto(resultados_vz, t_max_vz, texto_convergencia_vz, unidades=UNITS_ABS))

        st.subheader("Verificação de estabilidade dos pontos de equilíbrio")
        st.caption(
            "Análise numérica do modelo matemático para os parâmetros "
            "atualmente definidos, com as coordenadas dos pontos de "
            "equilíbrio já convertidas em número absoluto de indivíduos."
        )
        with st.expander("Ver verificação de estabilidade", expanded=False):
            st.markdown(relatorio_estabilidade(dim_vz, p_vz, fatores=fatores_abs_vz))

# ==========================================================================
# PÁGINA 3 — SIMULAÇÕES NUMÉRICAS
# ==========================================================================
else:
    st.title("Simulações Numéricas")

    st.markdown(
        "Preencha as condições iniciais e a área de cada população nos "
        "blocos abaixo, ajuste os demais parâmetros "
        "do modelo abaixo (a explicação de cada um está na página "
        "**Modelo Matemático**, no menu lateral) e clique em **Rodar "
        "simulação**."
    )
    st.markdown(
        "A capacidade suporte de cada espécie é calculada a "
        "partir da densidade máxima e da "
        "área informada. \\ Internamente, o "
        "sistema é sempre resolvido na forma **adimensional**. Os "
        "resultados são então apresentados em dois formatos: **adimensional** "
        "(cada população em relação à própria capacidade suporte, o que permite"
        " comparar as quatro espécies na mesma escala) e "
        "**dimensional** (a visão da trajetória em número absoluto de indivíduos)."
    )
    st.caption(
        "Os valores padrão preenchidos abaixo (densidades iniciais, áreas "
        "de habitat e demais parâmetros do modelo) correspondem ao estudo "
        "de caso da Represa de Várzea das Flores; para mais detalhes sobre "
        "a origem desses valores, confira a aba **Aplicação em Várzea das "
        "Flores**, no menu lateral."
    )

    st.subheader("Densidade inicial de cada população")
    modo_inicial = st.radio(
        "Como deseja informar a condição inicial?",
        [MODO_DENSIDADE, MODO_PERCENTUAL],
        horizontal=True, key="modo_inicial",
    )

    # densidade máxima (d_x) atualmente definida pelo usuário no expander
    # "Parâmetros do modelo" (abaixo) — lida via session_state porque, na
    # ordem do script, esse expander só é criado mais adiante; o valor
    # aqui já reflete a última interação do usuário (ou o padrão da
    # Tabela 2, se ele ainda não tiver mexido no campo).
    d_a_atual = st.session_state.get("d_a", DEFAULTS["k_a"])
    d_g_atual = st.session_state.get("d_g", DEFAULTS["k_g"])
    d_s_atual = st.session_state.get("d_s", DEFAULTS["k_s"])
    d_e_atual = st.session_state.get("d_e", DEFAULTS["k_e"])

    if modo_inicial == MODO_DENSIDADE:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            A0 = st.number_input("$A_0$: Aguapés (aguapés/m²)",
                                  min_value=0.0,
                                  value=st.session_state.get("A0", DENSIDADE_INICIAL_PADRAO["A"]),
                                  format="%.6g", key="A0")
            st.caption(f"≈ {A0 / d_a_atual * 100:.4g}% de $d_a$")
        with c2:
            G0 = st.number_input("$G_0$: Girinos (girinos/m²)",
                                  min_value=0.0,
                                  value=st.session_state.get("G0", DENSIDADE_INICIAL_PADRAO["G"]),
                                  format="%.6g", key="G0")
            st.caption(f"≈ {G0 / d_g_atual * 100:.4g}% de $d_g$")
        with c3:
            S0 = st.number_input("$S_0$: Sapos adultos (sapos/m²)",
                                  min_value=0.0,
                                  value=st.session_state.get("S0", DENSIDADE_INICIAL_PADRAO["S"]),
                                  format="%.6g", key="S0")
            st.caption(f"≈ {S0 / d_s_atual * 100:.4g}% de $d_s$")
        with c4:
            E0 = st.number_input("$E_0$: Escorpiões (escorpiões/m²)",
                                  min_value=0.0,
                                  value=st.session_state.get("E0", DENSIDADE_INICIAL_PADRAO["E"]),
                                  format="%.6g", key="E0")
            st.caption(f"≈ {E0 / d_e_atual * 100:.4g}% de $d_e$")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pct_A0 = st.number_input("$A_0$: Aguapés (% de $d_a$)",
                                      min_value=0.0, max_value=100.0,
                                      value=st.session_state.get("pct_A0", FRACOES_PADRAO_SIM["A"] * 100),
                                      format="%.4g", key="pct_A0")
            A0 = pct_A0 / 100 * d_a_atual
            st.caption(f"≈ {A0:.4g} aguapés/m²")
        with c2:
            pct_G0 = st.number_input("$G_0$: Girinos (% de $d_g$)",
                                      min_value=0.0, max_value=100.0,
                                      value=st.session_state.get("pct_G0", FRACOES_PADRAO_SIM["G"] * 100),
                                      format="%.4g", key="pct_G0")
            G0 = pct_G0 / 100 * d_g_atual
            st.caption(f"≈ {G0:.4g} girinos/m²")
        with c3:
            pct_S0 = st.number_input("$S_0$: Sapos adultos (% de $d_s$)",
                                      min_value=0.0, max_value=100.0,
                                      value=st.session_state.get("pct_S0", FRACOES_PADRAO_SIM["S"] * 100),
                                      format="%.4g", key="pct_S0")
            S0 = pct_S0 / 100 * d_s_atual
            st.caption(f"≈ {S0:.4g} sapos/m²")
        with c4:
            pct_E0 = st.number_input("$E_0$: Escorpiões (% de $d_e$)",
                                      min_value=0.0, max_value=100.0,
                                      value=st.session_state.get("pct_E0", FRACOES_PADRAO_SIM["E"] * 100),
                                      format="%.4g", key="pct_E0")
            E0 = pct_E0 / 100 * d_e_atual
            st.caption(f"≈ {E0:.4g} escorpiões/m²")

    st.subheader("Área total de habitat")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        r_a = st.number_input("$r_a$: Região de habitat de aguapés (m²)", min_value=1e-9,
                               value=AREAS_PADRAO_SIM["A"], format="%.6g", key="r_a")
    with a2:
        r_g = st.number_input("$r_g$: Região de habitat de girinos (m²)", min_value=1e-9,
                               value=AREAS_PADRAO_SIM["G"], format="%.6g", key="r_g")
    with a3:
        r_s = st.number_input("$r_s$: Região de habitat de sapos (m²)", min_value=1e-9,
                               value=AREAS_PADRAO_SIM["S"], format="%.6g", key="r_s")
    with a4:
        r_e = st.number_input("$r_e$: Região de habitat de escorpiões (m²)", min_value=1e-9,
                               value=AREAS_PADRAO_SIM["E"], format="%.6g", key="r_e")

    with st.expander("Sugestões de cenários iniciais"):

        nomes_cenarios = list(CENARIOS_INICIAIS.keys())

        for nome in nomes_cenarios:
            cenario = CENARIOS_INICIAIS[nome]
            st.markdown(f"**{nome}**")
            st.caption(cenario["desc"])
            st.button(
                "Usar este cenário",
                key=f"btn_{nome}",
                on_click=aplicar_cenario,
                args=(nome,),
                use_container_width=True,
            )

    with st.expander("Parâmetros do modelo"):
            st.caption(
                "$d_a, d_g, d_s, d_e$ são as densidades máximas de "
                "referência da Tabela 2 do artigo (indivíduos/m²); junto "
                "com a área $r_x$ definida acima, determinam a capacidade "
                "suporte $k_x = d_x \\times r_x$ de cada espécie. Os "
                "demais parâmetros ($\\alpha, \\theta$ incluídos) não "
                "reaparecem diretamente nos gráficos (ver explicação "
                "acima) — são usados apenas para calcular corretamente as "
                "razões adimensionais do modelo ($\\bar{k}, \\bar{\\alpha}, "
                "\\bar{\\beta}, \\bar{\\theta}$)."
            )
            p1, p2, p3 = st.columns(3)
            with p1:
                n_a = st.number_input(rotulo("n_a"), value=DEFAULTS["n_a"], format="%.6f")
                n_g = st.number_input(rotulo("n_g"), value=DEFAULTS["n_g"], format="%.6f")
                n_e = st.number_input(rotulo("n_e"), value=DEFAULTS["n_e"], format="%.6f")
                d_a = st.number_input("$d_a$ — Aguapés (aguapés/m²)", min_value=1e-12,
                                       value=DEFAULTS["k_a"], format="%.6g", key="d_a")
            with p2:
                d_g = st.number_input("$d_g$ — Girinos (girinos/m²)", min_value=1e-12,
                                       value=DEFAULTS["k_g"], format="%.6g", key="d_g")
                d_s = st.number_input("$d_s$ — Sapos adultos (sapos/m²)", min_value=1e-12,
                                       value=DEFAULTS["k_s"], format="%.6g", key="d_s")
                d_e = st.number_input("$d_e$ — Escorpiões (escorpiões/m²)", min_value=1e-12,
                                       value=DEFAULTS["k_e"], format="%.6g", key="d_e")
                mu_a = st.number_input(rotulo("mu_a"), value=DEFAULTS["mu_a"], format="%.6f")
            with p3:
                mu_s = st.number_input(rotulo("mu_s"), value=DEFAULTS["mu_s"], format="%.6f")
                delta = st.number_input(rotulo("delta"), value=DEFAULTS["delta"], format="%.6e")
                alpha = st.number_input(rotulo("alpha"), min_value=1e-9, value=DEFAULTS["alpha"], format="%.6f")
                beta = st.number_input(rotulo("beta"), value=DEFAULTS["beta"], format="%.6f")
            theta = st.number_input(rotulo("theta"), value=DEFAULTS["theta"], format="%.6f")

    # -- capacidade suporte k_x = d_x * r_x, e alpha/theta também
    #    convertidos para valores absolutos (alpha = d_alpha*r_a,
    #    theta = d_theta*r_e, conforme Tabela 2 do artigo). Esses valores
    #    absolutos são os que efetivamente entram nas equações do sistema
    #    dimensional (exibidos apenas após rodar a simulação, junto da
    #    contagem absoluta).
    areas_sim = {"A": r_a, "G": r_g, "S": r_s, "E": r_e}
    k_a_abs, k_g_abs, k_s_abs, k_e_abs = d_a * r_a, d_g * r_g, d_s * r_s, d_e * r_e
    alpha_abs = alpha * r_a
    theta_abs = theta * r_e

    st.subheader("Tempo de simulação")
    t_max = st.slider("Duração máxima da simulação (ano(s))", min_value=1, max_value=100, value=20)
    st.caption(
        "O gráfico mostrará o tempo de estabilização, "
        "definido como o instante a partir do qual a população "
        "permanece dentro de uma faixa de 2% em torno do valor final observado "
        "no período simulado. "
        "Se o período for curto demais, a estabilização pode não ter sido "
        "atingida, tente aumentar a duração da simulação."
    )

    # -- dim já com as capacidades suporte e alpha/theta em valores
    #    absolutos: é o que entra tanto na simulação dimensional quanto
    #    (via adimensionaliza_parametros) na análise de equilíbrio
    dim = dict(n_a=n_a, n_g=n_g, n_e=n_e, k_a=k_a_abs, k_g=k_g_abs, k_s=k_s_abs, k_e=k_e_abs,
               mu_a=mu_a, mu_s=mu_s, delta=delta, alpha=alpha_abs, beta=beta, theta=theta_abs)

    rodar = st.button("Rodar simulação", type="primary")

    if rodar:
        # -- validações básicas
        if n_a <= 0:
            st.error("n_a deve ser positivo (é usado para adimensionalizar o tempo).")
            st.stop()
        if d_a <= 0 or d_g <= 0 or d_s <= 0 or d_e <= 0:
            st.error("As densidades máximas (d_a, d_g, d_s, d_e) devem ser positivas.")
            st.stop()
        if r_a <= 0 or r_g <= 0 or r_s <= 0 or r_e <= 0:
            st.error("As áreas (r_a, r_g, r_s, r_e) devem ser positivas.")
            st.stop()

        # -- 1. converte parâmetros para adimensional. Como dim já usa as
        #      capacidades suporte absolutas (k_x_abs = d_x * r_x), p sai
        #      idêntico ao que se obteria com as densidades puras (a área
        #      se cancela nas razões), então serve tanto para a integração
        #      quanto para a análise de equilíbrio/estabilidade.
        p = adimensionaliza_parametros(dim)

        # -- 2. resolve o sistema ADIMENSIONAL: a densidade inicial
        #      informada é convertida para fração da densidade máxima
        #      (A0/d_a etc. — equivalente a dividir a quantidade absoluta
        #      A0*r_a pela capacidade suporte k_a_abs = d_a*r_a) e o tempo
        #      é reescalado por n_a.
        y0_bar = [A0 / d_a, G0 / d_g, S0 / d_s, E0 / d_e]
        t_bar_max = t_max * n_a
        t_bar_eval = np.linspace(0, t_bar_max, 3000)

        sol = solve_ivp(
            sistema_adimensional, [0, t_bar_max], y0_bar,
            args=(p,), t_eval=t_bar_eval, method="LSODA",
            rtol=1e-8, atol=1e-10,
        )

        if not sol.success:
            st.error(f"A integração numérica falhou: {sol.message}")
            st.stop()

        t_dim = sol.t / n_a
        ordem = ["A", "G", "S", "E"]

        # -- forma adimensional: cada espécie em relação à própria
        #    capacidade suporte (teto em 1)
        dados_rel = {"A": sol.y[0], "G": sol.y[1], "S": sol.y[2], "E": sol.y[3]}

        # -- forma dimensional (população absoluta): fração adimensional
        #    × capacidade suporte absoluta k_x_abs = d_x * r_x
        k_map = {"A": k_a_abs, "G": k_g_abs, "S": k_s_abs, "E": k_e_abs}
        dados_abs = {chave: dados_rel[chave] * k_map[chave] for chave in ordem}

        # o tempo de estabilização é o mesmo nas duas formas, pois escalar
        # por uma constante positiva não muda o critério relativo usado
        resultados = {chave: analisa_populacao(t_dim, dados_abs[chave]) for chave in ordem}

        # -- verificação de estabilidade dos pontos de equilíbrio, já nas
        #    mesmas unidades absolutas usadas no gráfico dimensional
        st.subheader("Verificação de estabilidade dos pontos de equilíbrio")
        st.caption("Análise numérica do modelo matemático para os parâmetros atualmente definidos.")
        with st.expander("Ver verificação de estabilidade", expanded=False):
            st.markdown(relatorio_estabilidade(dim, p))

        # -- gráficos, layout 1: forma adimensional (teto em 1)
        st.subheader("Resultados — forma adimensional")
        st.caption(
            "Cada gráfico expressa a população em relação à própria "
            "capacidade suporte da espécie, o que permite comparar "
            "visualmente a dinâmica das quatro espécies na mesma escala."
        )
        fig_rel, axs_rel = plt.subplots(2, 2, figsize=(12, 8))
        for ax, chave in zip(axs_rel.flat, ordem):
            _tend, _yf, t_estavel, idx_estavel, _var = resultados[chave]

            ax.plot(t_dim, dados_rel[chave], color=CORES[chave], linewidth=2)

            if idx_estavel is not None:
                t_marco = t_dim[idx_estavel]
                y_marco = dados_rel[chave][idx_estavel]
                ax.scatter([t_marco], [y_marco], color=CORES[chave], zorder=5, s=40)
                ax.axvline(t_marco, color=CORES[chave], linestyle="--", linewidth=1, alpha=0.4)
                ax.set_title(f"{NOMES[chave]} ({chave}) — estabilizou em {t_marco:.1f} ano(s)")
            else:
                ax.set_title(f"{NOMES[chave]} ({chave})")

            ax.set_xlabel("Tempo (ano(s))")
            ax.set_ylabel("População (forma adimensional)")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        fig_rel.tight_layout()
        st.pyplot(fig_rel)

        # -- gráficos, layout 2: forma dimensional (população absoluta)
        st.subheader("Resultados — forma dimensional (população absoluta)")
        st.caption(
            "Mesma trajetória acima, convertida em número absoluto de "
            "indivíduos; cada gráfico em uma escala "
            "própria, já que as capacidades suporte reais diferem."
        )
        fig_abs, axs_abs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, chave in zip(axs_abs.flat, ordem):
            _tend, yf, t_estavel, idx_estavel, _var = resultados[chave]

            ax.plot(t_dim, dados_abs[chave], color=CORES[chave], linewidth=2)

            if idx_estavel is not None:
                t_marco = t_dim[idx_estavel]
                y_marco = dados_abs[chave][idx_estavel]
                ax.scatter([t_marco], [y_marco], color=CORES[chave], zorder=5, s=40)
                ax.axvline(t_marco, color=CORES[chave], linestyle="--", linewidth=1, alpha=0.4)
                ax.set_title(f"{NOMES[chave]} ({chave}) — estabilizou em {t_marco:.1f} ano(s)")
            else:
                ax.set_title(f"{NOMES[chave]} ({chave})")

            ax.set_xlabel("Tempo (ano(s))")
            ax.set_ylabel("Indivíduos")
            ax.grid(alpha=0.3)
        fig_abs.tight_layout()
        st.pyplot(fig_abs)

        # -- análise textual automática, em número absoluto de indivíduos
        st.subheader("Interpretação dos resultados")
        texto_convergencia = identifica_convergencia(dim, p, dados_abs)
        st.markdown(gera_texto(resultados, t_max, texto_convergencia, unidades=UNITS_ABS))

        st.caption(
            f"Capacidade suporte calculada: "
            f"$k_a$=**{k_a_abs:,.4g}** aguapés,"
            f" $k_g$=**{k_g_abs:,.4g}** girinos,"
            f" $k_s$=**{k_s_abs:,.4g}** sapos,"
            f" $k_e$=**{k_e_abs:,.4g}** escorpiões."
        )
