# generate_dataset_kaggle.py
# Generates PT-PT Q&A pairs about Portugal using local Qwen2.5-7B inference.
# Runs entirely on Kaggle T4 GPU — no HF Inference API, no external API calls.
# Model is loaded once and reused for all batches.
#
# ── QUICK START ──────────────────────────────────────────────────────────────
#
# STEP 1 — TEST MODE (always run this first):
#   Set TEST_MODE = True below (it's the default).
#   Run the cell. It generates 10 pairs in ~2-5 min and saves to /kaggle/working/generated_pairs.json.
#   If you see "[0/10] ..." progress and a final "Done. 10 pairs saved" → full run is safe.
#
# STEP 2 — FULL RUN (only after test passes):
#   Set TEST_MODE = False, TARGET = 5000.
#   Run the cell. Progress is printed every batch.
#   When done (or when Kaggle session limit approaches), click "Save Version".
#
# STEP 3 — RESUME after a session ends:
#   After "Save Version", go to your dataset on Kaggle and note the version number.
#   In the next session, add the output dataset as an INPUT dataset.
#   Set RESUME_INPUT_PATH to the mounted path (e.g. "/kaggle/input/your-dataset/generated_pairs.json").
#   Run again — it will seed from the saved pairs and continue.
#
# Output: /kaggle/working/generated_pairs.json (incremental saves after every batch)
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Install dependencies ─────────────────────────────────────────────
# Pin exact versions to avoid the accelerate 1.12.0 circular import bug
# (accelerate.big_modeling ↔ accelerate.hooks circular dependency).
import subprocess
subprocess.run([
    "pip", "install", "--quiet", "--force-reinstall",
    "accelerate==0.34.2",
    "transformers==4.47.0",
    "bitsandbytes>=0.43.0",
], check=True)

# Reload transformers/accelerate after reinstall so the new versions are active.
# Do NOT evict numpy/scipy — reinstalling numpy mid-kernel corrupts its C extensions.
import importlib, sys
for mod in list(sys.modules.keys()):
    if mod.startswith("transformers") or mod.startswith("accelerate"):
        del sys.modules[mod]

# ── Imports ───────────────────────────────────────────────────────────────────
import importlib
import json
import os
import re
import random
import time
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# BitsAndBytesConfig import is deferred — we only use it if quantization works.
def _try_import_bnb_config():
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        return BitsAndBytesConfig
    except Exception as e:
        print(f"[WARN] bitsandbytes not available ({e}). Will use fp16 instead.", flush=True)
        return None

# ── Config ────────────────────────────────────────────────────────────────────
# Set TEST_MODE = True for a quick 10-pair sanity check before the full run.
TEST_MODE   = False         # ← CHANGE TO False FOR FULL RUN

TARGET      = 50 if TEST_MODE else 2000
BATCH_SIZE  = 5  if TEST_MODE else 5
FRESH_START = False  # ← True = always generate from scratch, ignoring any existing output file

GENERATOR_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE      = "/kaggle/working/generated_pairs.json"
MAX_NEW_TOKENS   = 1200 if TEST_MODE else 1500
TEMPERATURE      = 0.85

# Resume: set this to the path of a previously saved generated_pairs.json
# mounted as an input dataset, e.g. "/kaggle/input/YOUR-DATASET/generated_pairs.json"
RESUME_INPUT_PATH = "/kaggle/input/pt-qa-generated-1001/generated_pairs_1001.json"   # leave "" to start fresh

# ── VRAM / GPU check ─────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB", flush=True)

# ── Load existing instructions from the HF Hub base dataset ──────────────────
EXISTING_INSTRUCTIONS: set[str] = set()

try:
    from datasets import load_dataset
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    base_ds = load_dataset("nelsondiasandre/portuguese-qa-instruct-raw", token=hf_token)
    for split in base_ds.values():
        for ex in split:
            if "instruction" in ex:
                EXISTING_INSTRUCTIONS.add(ex["instruction"])
    print(f"Loaded {len(EXISTING_INSTRUCTIONS)} existing instructions from HF Hub.", flush=True)
except Exception as e:
    print(f"Could not load base dataset ({e}). Starting with empty duplicate set.", flush=True)

# ── Topic seeds — 100% Portugal ───────────────────────────────────────────────
TOPIC_SEEDS = [
    ("História Medieval de Portugal", [
        "fundação do condado portucalense e D. Afonso Henriques",
        "batalha de São Mamede (1128) e independência",
        "conquista de Lisboa aos mouros (1147)",
        "Reconquista cristã e expansão para o sul",
        "reinado de D. Afonso II e centralização do poder",
        "D. Dinis: o rei lavrador e a língua portuguesa",
        "crise de 1383-1385 e batalha de Aljubarrota",
        "fundação da dinastia de Avis por D. João I",
        "tratado de Windsor (1386) com a Inglaterra",
        "castelos medievais e arquitetura militar portuguesa",
    ]),
    ("Era dos Descobrimentos", [
        "Infante D. Henrique e a Escola de Sagres",
        "exploração da costa africana no século XV",
        "chegada à Índia por Vasco da Gama (1498)",
        "Pedro Álvares Cabral e a chegada ao Brasil (1500)",
        "tratado de Tordesilhas (1494) e divisão do mundo",
        "Fernão de Magalhães e a primeira volta ao mundo",
        "império português no século XVI: rotas e feitorias",
        "cartografia e instrumentos de navegação portugueses",
        "papel dos navegadores: Bartolomeu Dias, Gil Eanes",
        "impacto dos descobrimentos na Europa e no mundo",
    ]),
    ("Portugal nos séculos XVII–XIX", [
        "União Ibérica (1580–1640) e domínio filipino",
        "Restauração da Independência em 1640",
        "reinado de D. João IV e guerra da Restauração",
        "reinado de D. João V e o ouro do Brasil",
        "terramoto de Lisboa de 1755 e reconstrução pombalina",
        "Marquês de Pombal: reformas e iluminismo português",
        "invasões francesas (1807–1811) e guerra peninsular",
        "transferência da corte portuguesa para o Brasil (1807)",
        "liberalismo português e guerra civil (1828–1834)",
        "implantação da monarquia constitucional",
    ]),
    ("Primeira República e Estado Novo", [
        "implantação da República a 5 de outubro de 1910",
        "instabilidade política da Primeira República (1910–1926)",
        "golpe militar de 28 de maio de 1926",
        "ascensão de Salazar ao poder e o Estado Novo",
        "Constituição de 1933 e estrutura do regime",
        "PIDE: polícia política e repressão",
        "guerra colonial em Angola, Guiné e Moçambique",
        "oposição ao regime: Humberto Delgado, Álvaro Cunhal",
        "censura e propaganda no Estado Novo",
        "papel da Igreja Católica no Estado Novo",
    ]),
    ("Revolução de Abril e Democracia", [
        "25 de Abril de 1974: o golpe do MFA",
        "os capitães de Abril e o Movimento das Forças Armadas",
        "PREC: Processo Revolucionário em Curso (1974–1975)",
        "descolonização e independência das ex-colónias",
        "retornados: regresso dos portugueses das colónias",
        "Constituição de 1976 e instauração da democracia",
        "primeiros governos constitucionais e normalização",
        "adesão à CEE em 1986 e integração europeia",
        "presidência de Mário Soares e a consolidação democrática",
        "Portugal no século XXI: crise de 2011 e troika",
    ]),
    ("Geografia Física de Portugal Continental", [
        "principais rios portugueses: Tejo, Douro, Minho, Guadiana",
        "serras de Portugal Continental: Estrela, Gerês, Monchique",
        "regiões naturais: Minho, Trás-os-Montes, Beiras, Alentejo, Algarve",
        "costa atlântica: praias, falésias e estuários",
        "clima em Portugal: mediterrânico, atlântico, continental",
        "planície alentejana e agricultura de sequeiro",
        "vale do Douro e paisagem vitícola",
        "cabo de São Vicente e o sudoeste algarvio",
        "lagoas e zonas húmidas: Ria de Aveiro, Ria Formosa",
        "fronteira com Espanha e rios fronteiriços",
    ]),
    ("Açores e Madeira", [
        "formação geológica e vulcanismo dos Açores",
        "as 9 ilhas dos Açores: grupos oriental, central e ocidental",
        "ilha de São Miguel: caldeiras e termas",
        "ilha do Pico: montanha e vinha patrimônio UNESCO",
        "ilha das Flores e do Corvo: as mais ocidentais da Europa",
        "arquipélago da Madeira: história e colonização",
        "levadas da Madeira e sistema de irrigação",
        "Funchal: capital da Madeira e turismo",
        "produção de vinho Madeira e banana",
        "autonomia política dos Açores e da Madeira",
    ]),
    ("Cidades e Municípios de Portugal", [
        "Lisboa: história, bairros históricos e monumentos",
        "Porto: história, ribeira, vinho do Porto e ponte D. Luís",
        "Coimbra: universidade, fado de Coimbra e história",
        "Braga: arcebispado, Bom Jesus e Semana Santa",
        "Évora: templo romano, muralhas e património UNESCO",
        "Sintra: palácios românticos e património UNESCO",
        "Guimarães: berço de Portugal e centro histórico",
        "Faro e o Algarve: turismo, mar e interior",
        "Aveiro: moliceiros, canais e ovos moles",
        "Setúbal, Cascais, Viana do Castelo e outras cidades",
    ]),
    ("Reis e Rainhas de Portugal", [
        "D. Afonso Henriques: o fundador de Portugal",
        "D. Dinis: o rei poeta e administrador",
        "D. João I: fundador da dinastia de Avis",
        "D. Manuel I: o rei venturoso e o estilo manuelino",
        "D. João II: o príncipe perfeito",
        "D. Sebastião: o rei desejado e Alcácer Quibir (1578)",
        "D. João IV: restauração da independência",
        "D. João V: o rei magnânimo e o barroco português",
        "D. Maria I: a rainha piedosa e a viradeira",
        "D. Carlos I: o rei assassinado e o fim da monarquia",
    ]),
    ("Figuras da Cultura e das Letras", [
        "Luís de Camões: vida, obra e Os Lusíadas",
        "Fernando Pessoa e os seus heterónimos",
        "Eça de Queirós: realismo e crítica social",
        "José Saramago: Nobel da Literatura e obra",
        "Almeida Garrett: romantismo e liberalismo",
        "Alexandre Herculano: historiador e romancista",
        "Sophia de Mello Breyner Andresen: poetisa",
        "Gil Vicente: pai do teatro português",
        "Florbela Espanca: poetisa e vida trágica",
        "Almada Negreiros: modernismo e artes visuais",
    ]),
    ("Navegadores e Exploradores", [
        "Vasco da Gama: a rota para a Índia",
        "Pedro Álvares Cabral: chegada ao Brasil",
        "Bartolomeu Dias: o cabo da Boa Esperança (1488)",
        "Gil Eanes: passagem do cabo Bojador (1434)",
        "Fernão de Magalhães: primeira circum-navegação",
        "Diogo Cão: exploração do Congo e da costa africana",
        "Afonso de Albuquerque: governador da Índia portuguesa",
        "Duarte Pacheco Pereira: navegador e cartógrafo",
        "Infante D. Henrique: o mecenas dos descobrimentos",
        "impacto dos navegadores portugueses no comércio global",
    ]),
    ("Heróis e Figuras Políticas", [
        "Nuno Álvares Pereira: herói de Aljubarrota",
        "Aristides de Sousa Mendes: cônsul de Bordéus",
        "Humberto Delgado: o general sem medo",
        "Mário Soares: pai da democracia portuguesa",
        "Álvaro Cunhal: líder histórico do PCP",
        "Ramalho Eanes: primeiro presidente eleito democraticamente",
        "Aníbal Cavaco Silva: reformas económicas dos anos 80-90",
        "António de Oliveira Salazar: figura e legado controverso",
        "Marcelo Caetano: o sucessor de Salazar",
        "Otelo Saraiva de Carvalho: figura do 25 de Abril",
    ]),
    ("Fado e Música Portuguesa", [
        "origens do fado: mito e história",
        "fado de Lisboa vs fado de Coimbra",
        "Amália Rodrigues: vida e legado",
        "Mariza, Cristina Branco e o novo fado",
        "instrumentos do fado: guitarra portuguesa e viola baixo",
        "fado como Património Imaterial da UNESCO",
        "música popular portuguesa: pimba e outros géneros",
        "rock português: Xutos & Pontapés, GNR, Rui Veloso",
        "música tradicional regional: viras, chulas, corridinhos",
        "Portugal no Festival da Eurovisão",
    ]),
    ("Literatura Portuguesa", [
        "trovadorismo: cantigas de amigo, amor e escárnio",
        "humanismo português do século XVI",
        "Os Lusíadas de Camões: estrutura e temas",
        "Romantismo português: Garrett e Herculano",
        "Realismo e Naturalismo: Eça de Queirós e Cesário Verde",
        "Modernismo: Pessoa, Sá-Carneiro, Almada Negreiros",
        "Neorrealismo: Alves Redol e a literatura operária",
        "literatura portuguesa pós-25 de Abril",
        "poesia portuguesa contemporânea",
        "Agustina Bessa-Luís e a prosa do século XX",
    ]),
    ("Arte e Arquitetura Portuguesa", [
        "arte românica em Portugal: Sé de Coimbra, Sé do Porto",
        "arte gótica: Batalha, Alcobaça, mosteiros",
        "estilo manuelino: Jerónimos, Torre de Belém",
        "azulejo português: história e evolução",
        "pintura portuguesa: Nuno Gonçalves e os painéis de S. Vicente",
        "barroco português: Mafra, talha dourada",
        "modernismo arquitetónico: Álvaro Siza Vieira",
        "design português contemporâneo",
        "museus em Portugal: Gulbenkian, MAAT, Arte Antiga",
        "arte urbana e grafiti em Lisboa e Porto",
    ]),
    ("Tradições, Festas e Folclore", [
        "Santos Populares: Santo António, São João, São Pedro",
        "arraiais e marchas populares de Lisboa",
        "Festa de São João no Porto",
        "Semana Santa em Braga e Ovar",
        "Festa dos Tabuleiros em Tomar",
        "Carnaval de Torres Vedras e Loulé",
        "romarias: Nossa Senhora da Agonia, Fátima, Bom Jesus",
        "traje regional: minhoto, algarvio, ribatejano",
        "gaitas-de-foles e música tradicional do Minho",
        "lendas e mitos portugueses: D. Sebastião, Inês de Castro",
    ]),
    ("Gastronomia e Vinhos de Portugal", [
        "bacalhau: história e as 365 receitas",
        "pastel de nata e doçaria de Belém",
        "gastronomia do Norte: caldo verde, rojões, francesinha",
        "gastronomia do Alentejo: açorda, migas, carne de porco",
        "gastronomia do Algarve: cataplana, percebes, amêijoas",
        "doçaria conventual: ovos moles, pastéis de Tentúgal",
        "vinho do Porto: história, produção e estilos",
        "vinho verde: região e características",
        "vinhos do Alentejo, Dão e Bairrada",
        "azeite português e a olivicultura tradicional",
    ]),
    ("Futebol Português", [
        "história do futebol em Portugal",
        "Benfica: história, títulos e grandes jogadores",
        "FC Porto: história, títulos e Liga dos Campeões de 2004",
        "Sporting CP: história e títulos",
        "seleção nacional: Europeu de 2016, Liga das Nações",
        "Cristiano Ronaldo: carreira e recordes",
        "Eusébio: o Pantera Negra e o Mundial de 1966",
        "Luís Figo: Bola de Ouro e carreira internacional",
        "outros grandes futebolistas portugueses do século XX-XXI",
        "estádios portugueses e a Euro 2004",
    ]),
    ("Outros Desportos Portugueses", [
        "hóquei em patins: domínio mundial português",
        "atletismo: Rosa Mota, Carlos Lopes, Fernando Mamede",
        "ciclismo: Volta a Portugal e ciclistas portugueses",
        "surf em Portugal: Nazaré e as ondas gigantes",
        "ténis: João Sousa e outros tenistas portugueses",
        "andebol e basquetebol em Portugal",
        "vela e desportos náuticos portugueses",
        "automobilismo: pilotos e ralis portugueses",
        "Portugal nos Jogos Olímpicos: medalhas e atletas",
        "tourada portuguesa: tradição e controvérsia",
    ]),
    ("Sistema Político Português", [
        "Constituição da República Portuguesa de 1976",
        "semipresidencialismo português: poderes do Presidente",
        "Assembleia da República: composição e funções",
        "partidos políticos portugueses: PS, PSD, Chega, BE, PCP",
        "sistema eleitoral e o método de Hondt",
        "Tribunal Constitucional e fiscalização das leis",
        "poder local: câmaras municipais e juntas de freguesia",
        "regiões autónomas: Açores e Madeira",
        "provedoria de justiça e direitos dos cidadãos",
        "relações entre Presidente, Governo e Parlamento",
    ]),
    ("Portugal e o Mundo", [
        "Portugal na NATO: membro fundador (1949)",
        "adesão à CEE/UE em 1986 e integração europeia",
        "CPLP: Comunidade dos Países de Língua Portuguesa",
        "relações luso-brasileiras: história e atualidade",
        "relações luso-espanholas: história e cooperação",
        "aliança luso-britânica: a mais antiga do mundo",
        "papel de Portugal nas missões de paz da ONU",
        "Portugal e os PALOP: cooperação pós-colonial",
        "influência portuguesa em Timor-Leste",
        "emigração portuguesa e as comunidades no estrangeiro",
    ]),
    ("Economia Portuguesa", [
        "evolução económica de Portugal no século XX",
        "crise financeira de 2011 e intervenção da troika",
        "setor do turismo: impacto e crescimento pós-2010",
        "setor da cortiça: Portugal líder mundial",
        "indústria automóvel: AutoEuropa e fornecedores",
        "pescas e aquacultura em Portugal",
        "agricultura: vinho, azeite, frutas e horticultura",
        "startups e ecossistema tecnológico português",
        "construção e mercado imobiliário em Lisboa e Porto",
        "fundos europeus e Portugal 2030",
    ]),
    ("Natureza e Ambiente em Portugal", [
        "Parque Nacional da Peneda-Gerês",
        "Parque Natural da Ria Formosa",
        "Serra da Estrela: glaciação e biodiversidade",
        "sobreiro e a indústria da cortiça",
        "lince ibérico: reintrodução em Portugal",
        "lobo ibérico: proteção e conflito com pastores",
        "incêndios florestais em Portugal: causas e consequências",
        "energia eólica e solar em Portugal",
        "qualidade das praias portuguesas: bandeiras azuis",
        "rio Tejo: biodiversidade e gestão ambiental",
    ]),
    ("Religião e Espiritualidade em Portugal", [
        "catolicismo em Portugal: história e presença atual",
        "Santuário de Fátima: aparições de 1917 e peregrinações",
        "Inquisição portuguesa: criação, funcionamento e fim",
        "mouros e cristãos: convivência e reconquista",
        "judeus em Portugal: sefarditas e diáspora",
        "ordens religiosas em Portugal: Templários, Franciscanos",
        "mosteiros e conventos: Alcobaça, Batalha, Jerónimos",
        "festas religiosas populares: Páscoa, Natal, São João",
        "sincretismo e religiosidade popular portuguesa",
        "laicidade e novas religiões em Portugal contemporâneo",
    ]),
    ("Língua Portuguesa em Portugal", [
        "origem e evolução do português a partir do latim",
        "galego-português medieval e a sua separação",
        "influências árabes no português: palavras de origem árabe",
        "dialetos regionais: transmontano, alentejano, algarvio",
        "mirandês: língua minoritária de Trás-os-Montes",
        "Acordo Ortográfico de 1990 e polémicas",
        "expressões idiomáticas tipicamente portuguesas",
        "provérbios portugueses e a sua sabedoria popular",
        "diferenças entre PT-PT e PT-BR: léxico e pronúncia",
        "português como língua global e a lusofonia",
    ]),
    ("Ciência e Inovação em Portugal", [
        "universidades portuguesas e investigação científica",
        "contribuições portuguesas para a ciência mundial",
        "Nobel de Medicina de Egas Moniz: contexto e impacto",
        "Pedro Nunes: matemático e o nónio",
        "Garcia de Orta: médico e botânico do século XVI",
        "energias renováveis: Portugal e a meta de 100% renovável",
        "startups deeptech e biotech portuguesas",
        "Web Summit em Lisboa: impacto no ecossistema",
        "IPMA: meteorologia e oceanografia em Portugal",
        "António Damásio: neurocientista português",
    ]),
    ("Saúde em Portugal", [
        "SNS: criação em 1979 e evolução até hoje",
        "centros de saúde e medicina de família",
        "principais hospitais: Santa Maria, São João",
        "doenças mais prevalentes em Portugal",
        "envelhecimento e cuidados continuados",
        "saúde mental: situação e serviços em Portugal",
        "impacto da pandemia COVID-19 em Portugal",
        "vacinação em Portugal: plano nacional de vacinação",
        "medicina tradicional portuguesa e plantas medicinais",
        "investigação médica em Portugal",
    ]),
    ("Educação em Portugal", [
        "Universidade de Coimbra: a mais antiga (1290)",
        "sistema de ensino básico e secundário português",
        "acesso ao ensino superior: exames nacionais",
        "ensino profissional e formação vocacional",
        "taxa de analfabetismo em Portugal: evolução histórica",
        "programa Erasmus e mobilidade estudantil portuguesa",
        "ensino do português no estrangeiro",
        "Web Summit e literacia digital em Portugal",
        "Técnico Lisboa e as melhores universidades",
        "desafios do sistema educativo português atual",
    ]),
    ("Património da Humanidade em Portugal", [
        "Centro Histórico de Évora (1986)",
        "Mosteiro dos Jerónimos e Torre de Belém (1983)",
        "Mosteiro da Batalha (1983)",
        "Convento de Cristo em Tomar (1983)",
        "Centro Histórico do Porto (1996)",
        "Alto Douro Vinhateiro (2001)",
        "Centro Histórico de Guimarães (2001)",
        "Paisagem da Cultura da Vinha da Ilha do Pico (2004)",
        "Universidade de Coimbra — Alta e Sofia (2013)",
        "fado como Património Imaterial da Humanidade (2011)",
    ]),
    ("Origens de Portugal: Pré-história e Ibéria Romana", [
        "etimologia do nome 'Portugal': origem e significado",
        "povos pré-romanos: celtas, lusitanos e outros",
        "Viriato: líder lusitano e resistência a Roma",
        "romanização da Lusitânia: cidades, língua, cultura",
        "Olisipo (Lisboa), Ebora (Évora) e Bracara (Braga) romanas",
        "invasões germânicas: visigodos e suevos",
        "Ibéria muçulmana: al-Andalus e herança islâmica",
        "condado portucalense: origem do território português",
        "influências árabes na cultura e língua portuguesa",
        "monumentos megalíticos e castros em Portugal",
    ]),
    ("Forças Armadas e Defesa de Portugal", [
        "história das Forças Armadas portuguesas",
        "Portugal na NATO: contribuições e missões",
        "guerra colonial (1961–1974): Angola, Moçambique, Guiné",
        "papel do MFA na Revolução de Abril",
        "missões de paz: Timor, Bósnia, Afeganistão",
        "GNR e PSP: forças de segurança em Portugal",
        "serviço militar em Portugal: obrigatoriedade e reforma",
        "Armada Portuguesa: história e papel atual",
        "Força Aérea Portuguesa: história e meios",
        "Regimento de Cavalaria e tradições militares",
    ]),
    ("Transportes e Infraestrutura em Portugal", [
        "rede de autoestradas em Portugal: história e expansão",
        "CP (Comboios de Portugal): história e rede ferroviária",
        "Metro de Lisboa e Metro do Porto",
        "porto de Sines: o maior porto industrial de Portugal",
        "aeroporto Humberto Delgado: o principal de Portugal",
        "ponte 25 de Abril e ponte Vasco da Gama",
        "TAP Air Portugal: história e importância estratégica",
        "mobilidade urbana em Lisboa e Porto",
        "transportes nas ilhas: Açores e Madeira",
        "projeto de alta velocidade ferroviária",
    ]),
    ("Media e Comunicação em Portugal", [
        "RTP: televisão pública portuguesa e história",
        "imprensa portuguesa: Público, Expresso, Jornal de Notícias",
        "rádio em Portugal: história e principais emissoras",
        "liberdade de imprensa em Portugal pós-25 de Abril",
        "cinema português: realizadores e filmes emblemáticos",
        "redes sociais e media digital em Portugal",
        "Agência Lusa: agência noticiosa nacional",
        "publicidade e indústria criativa portuguesa",
        "jornalismo de investigação português",
        "censura no Estado Novo e liberdade de imprensa atual",
    ]),
    ("Feriados e Datas Cívicas de Portugal", [
        "25 de Abril: Dia da Liberdade e da Revolução",
        "10 de Junho: Dia de Portugal, de Camões e das Comunidades",
        "5 de Outubro: Implantação da República",
        "1 de Dezembro: Restauração da Independência",
        "significado histórico dos feriados nacionais portugueses",
        "feriados religiosos em Portugal: Páscoa, Natal, Assunção",
        "feriados municipais e tradições locais",
        "comemorações do 25 de Abril ao longo dos anos",
        "Dia de Camões e a identidade cultural portuguesa",
        "evolução dos feriados nacionais desde 1974",
    ]),
    ("Organização Administrativa de Portugal", [
        "os 18 distritos de Portugal Continental",
        "os 308 municípios de Portugal",
        "juntas de freguesia: a unidade administrativa básica",
        "área metropolitana de Lisboa: concelhos e população",
        "área metropolitana do Porto: concelhos e população",
        "regiões autónomas: autonomia dos Açores e Madeira",
        "reforma administrativa de 2013 e fusão de freguesias",
        "CCDR: Comissões de Coordenação Regional",
        "diferenças regionais: litoral vs interior em Portugal",
        "NUTS: regiões estatísticas de Portugal",
    ]),
]

# ── Persona & format templates ────────────────────────────────────────────────
_PERSONA = (
    "És um historiador e especialista em Portugal com conhecimento enciclopédico "
    "sobre todos os aspectos do país: história, geografia, cultura, pessoas notáveis, "
    "desporto, política, economia, gastronomia, arte, tradições e língua. "
    "Respondes SEMPRE em português europeu (PT-PT) com rigor académico mas linguagem acessível. "
    "NUNCA uses brasileirismos: proibido usar você, vocês, ônibus, celular, time (usa equipa), "
    "legal (usa fixe/óptimo), cadastrar, deletar, planilha, bilhões, trem, banheiro. "
    "Usa sempre formas europeias: tu/vós, autocarro, telemóvel, equipa, registar, eliminar, "
    "folha de cálculo, mil milhões, óptimo, frigorífico, pequeno-almoço."
)

FORMAT_TEMPLATES = [
    {
        "name": "factual",
        "weight": 15,
        "user": (
            "Gera {n} pares pergunta/resposta sobre Portugal, especificamente: "
            "{topic} — {subtopic}.\n"
            "Perguntas sobre Portugal especificamente. "
            "Varia o início: O que foi, Qual é, Quem foi, Quando, Onde fica, etc.\n"
            "Cada resposta: 2-4 frases, factual, com datas/nomes/locais concretos.\n"
            "Devolve APENAS JSON válido, sem texto antes ou depois:\n"
            "[\n"
            "  {{\"instruction\": \"Quando foi fundado o Mosteiro dos Jerónimos?\", "
            "\"response\": \"O Mosteiro dos Jerónimos foi mandado construir pelo rei D. Manuel I em 1501, em Belém, Lisboa. A sua construção demorou cerca de um século e representa o auge do estilo manuelino.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "correction",
        "weight": 15,
        "user": (
            "Gera {n} pares de correcção de mitos/erros sobre Portugal: "
            "{topic} — {subtopic}.\n"
            "A pergunta apresenta uma afirmação ERRADA sobre Portugal. "
            "A resposta começa com 'Não,' ou 'Isso é um equívoco.' ou 'Na verdade,' "
            "e corrige com factos precisos.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"O bacalhau é um peixe tipicamente português, correcto?\", "
            "\"response\": \"Isso é um equívoco. O bacalhau não é pescado em Portugal: vem sobretudo da Noruega e da Islândia. O que é tipicamente português é a sua preparação culinária, com centenas de receitas tradicionais.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "comparison",
        "weight": 10,
        "user": (
            "Gera {n} pares de comparação sobre Portugal: {topic} — {subtopic}.\n"
            "Compara dois aspectos, períodos, figuras ou regiões portuguesas entre si.\n"
            "Estrutura clara com factos concretos sobre Portugal.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Qual a diferença entre o fado de Lisboa e o fado de Coimbra?\", "
            "\"response\": \"O fado de Lisboa é cantado por homens e mulheres, com temática urbana e saudosista. O fado de Coimbra é exclusivamente masculino, associado à vida académica, com um tom mais introspectivo e poético.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "howto",
        "weight": 8,
        "user": (
            "Gera {n} pares 'como funciona/como se faz' sobre Portugal: "
            "{topic} — {subtopic}.\n"
            "Foca em processos, instituições ou tradições PORTUGUESAS específicas. "
            "A resposta deve mencionar Portugal explicitamente.\n"
            "Começa com: 'Como funciona', 'Como se realiza', 'De que forma'.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Como funciona o sistema eleitoral português?\", "
            "\"response\": \"Portugal usa o método de Hondt para converter votos em mandatos parlamentares. Os cidadãos votam em listas partidárias por círculo eleitoral, e os mandatos são distribuídos proporcionalmente aos votos recebidos por cada partido.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "why_reasoning",
        "weight": 12,
        "user": (
            "Gera {n} pares de raciocínio causal sobre Portugal: {topic} — {subtopic}.\n"
            "Explica PORQUE aconteceu algo em Portugal, PORQUE uma tradição existe.\n"
            "Começa com: 'Por que razão', 'Porque é que', 'O que levou Portugal a'.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Por que razão Portugal foi o primeiro país a iniciar os Descobrimentos marítimos?\", "
            "\"response\": \"Portugal beneficiou de uma posição geográfica privilegiada na fachada atlântica da Península Ibérica e de uma longa tradição marítima. A centralização do poder pela dinastia de Avis e o apoio do Infante D. Henrique à investigação náutica foram determinantes para o avanço das explorações a partir de 1415.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "enumeration",
        "weight": 10,
        "user": (
            "Gera {n} pares de enumeração sobre Portugal: {topic} — {subtopic}.\n"
            "Usa: 'Quais são os principais', 'Enumera os', 'Que X portugueses existem'.\n"
            "Resposta: lista numerada com 3-6 itens reais, cada um com breve explicação concreta.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Quais são os principais rios de Portugal Continental?\", "
            "\"response\": \"1. Tejo: o maior rio da Península Ibérica, nasce em Espanha e desagua em Lisboa. 2. Douro: percorre o norte do país e a sua bacia é a região do vinho do Porto. 3. Minho: forma a fronteira natural com a Galiza. 4. Guadiana: delimita parte da fronteira sul com Espanha.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "definition_example",
        "weight": 8,
        "user": (
            "Gera {n} pares definição+exemplo sobre Portugal: {topic} — {subtopic}.\n"
            "A pergunta pede definição de um conceito português E um exemplo concreto.\n"
            "Resposta: definição precisa + exemplo real português com nome, local ou data.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"O que é o estilo manuelino? Dá um exemplo.\", "
            "\"response\": \"O manuelino é um estilo arquitectónico português do início do século XVI, caracterizado por decoração exuberante com motivos marinhos, como cordas, esferas armilares e elementos naturais. O exemplo mais notável é o Mosteiro dos Jerónimos, em Belém, construído durante o reinado de D. Manuel I.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "debate",
        "weight": 10,
        "user": (
            "Gera {n} pares de análise crítica sobre Portugal: {topic} — {subtopic}.\n"
            "Apresenta afirmações simplistas ou controversas REAIS sobre {subtopic} "
            "e analisa-as com nuance. Exemplo: 'É correcto afirmar que Salazar modernizou Portugal?'\n"
            "Resposta nuançada: confirma o que é verdade, corrige o simplista, "
            "acrescenta contexto histórico ou social português.\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"É correcto afirmar que o Marquês de Pombal foi um ditador?\", "
            "\"response\": \"É uma simplificação. Pombal governou de forma autoritária e expulsou os Jesuítas em 1759, mas conduziu reformas essenciais: reconstruiu Lisboa após o terramoto de 1755, modernizou a economia e reformou o ensino. O seu legado é complexo — repressivo nos métodos, mas decisivo para a modernização de Portugal.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "contextual",
        "weight": 12,
        "user": (
            "Gera {n} pares contextuais sobre Portugal: {topic} — {subtopic}.\n"
            "Parte de factos históricos, geográficos ou culturais portugueses e explora "
            "o seu impacto ou legado. Exemplo: 'Qual o impacto do terramoto de 1755 em Lisboa?'\n"
            "Devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Qual o impacto do terramoto de 1755 em Lisboa?\", "
            "\"response\": \"O terramoto de 1 de Novembro de 1755 destruiu grande parte de Lisboa e matou entre 30 000 e 40 000 pessoas. O Marquês de Pombal liderou a reconstrução, criando a Baixa Pombalina com traçado moderno e anti-sísmico. O evento acelerou as reformas iluministas e enfraqueceu a influência da Igreja e da nobreza tradicional.\"}}\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
]

FORMAT_POOL = []
for fmt in FORMAT_TEMPLATES:
    FORMAT_POOL.extend([fmt] * fmt["weight"])

# ── Validation patterns ───────────────────────────────────────────────────────
_PTBR_PATTERN = re.compile(
    r"\bvocê\b|\bvocês\b|\bônibus\b|\bcelular\b|\btime\b|\blegal\b|"
    r"\bcadastrar\b|\bdeletar\b|\bplanilha\b|\bbilhões\b|\btrem\b|\bbanheiro\b|\bsobrenome\b",
    re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(
    r"\(pergunta real|\(item real|\(afirma.{1,4}o (real|controversa real)|"
    r"\(facto real|\(conceito real|\(resposta real|\(defini.{1,4}o real|\(an.lise (com|nuan)|"
    r"\[item\]|\[explica.{1,4}o\]|\[A\]\s*(e|em|vs)\s*\[B\]|"
    r"\[Pergunta sobre Portugal|\[Resposta factual|\[Afirma.{1,4}o errada|"
    r"\[Compara.{1,4}o|\[Explica.{1,4}o com detalhes|\[Explica.{1,4}o causal|"
    r"\[Defini.{1,4}o\]|\[exemplo real portugu|\[An.lise nuan|\[An.lise do impacto|"
    r"\[correc.{1,4}o com factos|\[processo portugu|\[itens portugu|"
    r"\[facto portugu|\[afirma.{1,4}o sobre Portugal|afirmar que \[|raz.o \[|"
    r"^\s*\[[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][^\]]{5,}\]\s*[.\n]",
    re.IGNORECASE | re.MULTILINE,
)

_OFFTRACK_RE = re.compile(r"^[^\n]{5,200}\?$")

_PORTUGAL_RE = re.compile(
    r"portugu[eê]|portugal|lisboa|porto\b|algarve|alentejo|bragan.a|"
    r"coimbra|fado\b|descobrimentos|sal.zar|rep.blica|av.s|lusit.n|lu.ofon|atl.ntico|ib.r|"
    r"afonso henriques|d\. afonso|d\. jo.o|d\. manuel|d\. dinis|d\. sebasti|"
    r"d\. carlos|d\. maria|d\. pedro|d\. fil.p|"
    r"diogo c.o|vasco da gama|bartolomeu dias|cabral|cam.es|pessoa\b|saramago|"
    r"salazar|pombal|e.a de queir|egas moniz|humberto delgado|m.rio soares|"
    r"aljubarrota|alc.cer quibir|tordesilhas|sagres|"
    r"guimar.es|braga\b|.vora|sintra|.bidos|tomar\b|batalha\b|alco.a.a|jer.nimos|"
    r"bel.m\b|madeira\b|a.ores\b|douro\b|tejo\b|minho\b|guadiana|"
    r"snc\b|sns\b|rtp\b|pide\b|mfa\b|prec\b|cplp\b|nato\b|cee\b|"
    r"reconquista|inquisi..o|estado novo|primeira rep.blica|"
    r"25 de abril|5 de outubro|10 de junho|1 de dezembro",
    re.IGNORECASE,
)


def contains_ptbr(text: str) -> bool:
    return bool(_PTBR_PATTERN.search(text))


def is_valid_pair(instr: str, resp: str, existing: set) -> bool:
    if len(instr) < 25 or len(resp) < 40:
        return False
    if instr in existing:
        return False
    if _PLACEHOLDER_RE.search(instr) or _PLACEHOLDER_RE.search(resp):
        return False
    if _OFFTRACK_RE.match(resp):
        return False
    if not _PORTUGAL_RE.search(instr + " " + resp):
        return False
    if contains_ptbr(instr) or contains_ptbr(resp):
        return False
    return True

# ── Load model — 4-bit quantization preferred, fp16 fallback ─────────────────
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)

BitsAndBytesConfig = _try_import_bnb_config()

if BitsAndBytesConfig is not None:
    print(f"\nLoading {GENERATOR_MODEL} with 4-bit NF4 quantization...", flush=True)
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Loaded with 4-bit quantization.", flush=True)
    except Exception as e:
        print(f"[WARN] 4-bit load failed ({e}). Falling back to fp16...", flush=True)
        BitsAndBytesConfig = None  # trigger fp16 path below

if BitsAndBytesConfig is None:
    # fp16 — ~14 GB on T4; may be tight but usually works
    print(f"\nLoading {GENERATOR_MODEL} in fp16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Loaded in fp16.", flush=True)

model.eval()
if torch.cuda.is_available():
    print(f"VRAM used after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# ── Generation helpers ────────────────────────────────────────────────────────
def generate_local(system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_input_tokens = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_new = outputs[0].shape[0] - n_input_tokens
    elapsed = time.time() - t0
    print(f"  [gen: {n_new} tokens in {elapsed:.1f}s = {n_new/elapsed:.1f} tok/s]", flush=True)
    new_tokens = outputs[0][n_input_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(text: str) -> list[dict] | None:
    text = text.strip()
    # Fast path: clean JSON
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Find outermost array
    start = text.find("[")
    if start == -1:
        return None
    end = text.rfind("]")
    if end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Truncated output: model was cut off before closing ].
    # Salvage complete objects by scanning for },{...} boundaries.
    chunk = text[start:]
    salvaged = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(chunk):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(chunk[obj_start:i + 1])
                    if isinstance(obj, dict):
                        salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
    if salvaged:
        print(f"  [JSON truncated — salvaged {len(salvaged)} complete object(s)]", flush=True)
        return salvaged
    return None


def generate_batch(topic, subtopic, fmt, n, existing):
    sample = list(existing)[-5:] if existing else []
    existing_sample = "; ".join(f'"{s[:50]}"' for s in sample) or "nenhuma"
    user_prompt = fmt["user"].format(
        topic=topic, subtopic=subtopic, n=n, existing_sample=existing_sample
    )
    try:
        raw = generate_local(_PERSONA, user_prompt)
    except Exception:
        print("  [generate_local exception]", flush=True)
        traceback.print_exc()
        return []

    pairs = extract_json(raw)
    if pairs is None:
        print(f"  [JSON parse failed — raw output snippet: {raw[:200]!r}]", flush=True)
        return []

    valid = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        instr = p.get("instruction", "").strip()
        resp  = p.get("response",    "").strip()
        if not is_valid_pair(instr, resp, existing):
            if instr and resp:
                print(f"  [rejected] {instr[:60]}", flush=True)
            continue
        valid.append({"instruction": instr, "response": resp})
        existing.add(instr)
    return valid

# ── Resume: seed from a previously saved run mounted as input ─────────────────
results: list[dict] = []

if not FRESH_START and RESUME_INPUT_PATH and os.path.exists(RESUME_INPUT_PATH):
    results = json.loads(open(RESUME_INPUT_PATH, encoding="utf-8").read())
    for p in results:
        EXISTING_INSTRUCTIONS.add(p["instruction"])
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resumed from input: {len(results)} pairs loaded.", flush=True)
elif not FRESH_START and os.path.exists(OUTPUT_FILE):
    results = json.loads(open(OUTPUT_FILE, encoding="utf-8").read())
    for p in results:
        EXISTING_INSTRUCTIONS.add(p["instruction"])
    print(f"Resumed from working dir: {len(results)} pairs already generated.", flush=True)
else:
    print("Fresh start — ignoring any existing output file.", flush=True)

# ── Main generation loop ──────────────────────────────────────────────────────
import datetime

combos = [(cat, sub) for cat, subs in TOPIC_SEEDS for sub in subs]
random.shuffle(combos)
combo_idx = 0

run_start = time.time()
last_milestone = len(results)  # last count at which we printed a milestone

def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

print(f"\nStart: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Target: {TARGET} | Starting from: {len(results)}", flush=True)
print(f"Topics: {len(TOPIC_SEEDS)} categories, {len(combos)} subtopics\n", flush=True)

while len(results) < TARGET:
    category, subtopic = combos[combo_idx % len(combos)]
    combo_idx += 1
    fmt = random.choice(FORMAT_POOL)
    n = min(BATCH_SIZE, TARGET - len(results))

    print(f"[{len(results)}/{TARGET}] {fmt['name']:20s} | {category[:35]} → {subtopic[:40]}", flush=True)

    pairs = generate_batch(category, subtopic, fmt, n, EXISTING_INSTRUCTIONS)

    if pairs:
        results.extend(pairs)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  +{len(pairs)} pairs (total: {len(results)})", flush=True)

        # Print elapsed time every 100 pairs
        if len(results) // 100 > last_milestone // 100:
            elapsed = time.time() - run_start
            rate = (len(results) - (TARGET - TARGET)) / elapsed if elapsed > 0 else 0
            print(f"\n── {len(results)} pairs ── elapsed: {fmt_elapsed(elapsed)} ──\n", flush=True)
            last_milestone = len(results)
    else:
        print("  (0 valid pairs, skipping)", flush=True)

end_time = datetime.datetime.now()
total_elapsed = time.time() - run_start
print(f"\nEnd:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"Elapsed: {fmt_elapsed(total_elapsed)}", flush=True)
print(f"Done. {len(results)} pairs saved to {OUTPUT_FILE}", flush=True)
print("Download from Kaggle Output panel → working/generated_pairs.json", flush=True)
