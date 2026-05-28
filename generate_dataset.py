# generate_dataset.py
# Generates PT-PT Q&A pairs via LM Studio local API (Qwen2.5-7B or similar).
# Goal: train the model to become a deep expert on PORTUGAL — history, geography,
# people, culture, sport, politics, economy, science, art, traditions, language.
# Produces varied answer formats: factual, correction, comparison, how-to,
# why/reasoning, enumeration, definition+example, debate, contextual.
#
# Usage:
#   python generate_dataset.py --target 5000 --batch 10 --out generated_pairs.json
#   python generate_dataset.py --target 5000 --batch 10 --out generated_pairs.json --resume
#
# After reviewing generated_pairs.json, run:
#   python generate_dataset.py --merge
# to merge into create_dataset.py and rebuild the dataset.

import argparse
import json
import os
import re
import time
import random
import urllib.request
import urllib.error

# ── LM Studio config ──────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID       = "qwen2.5-7b-instruct"  # adjust to match your LM Studio model name

# ── Topic seeds — 100% Portugal-focused ──────────────────────────────────────
# Every subtopic is a specific angle on Portugal. The model should become a
# Portuguese historian/expert that knows everything about Portugal.
TOPIC_SEEDS = [

    # ── HISTÓRIA ──────────────────────────────────────────────────────────────
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

    # ── GEOGRAFIA ─────────────────────────────────────────────────────────────
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
    ("Demografia e Sociedade Portuguesa", [
        "evolução da população portuguesa ao longo dos séculos",
        "emigração portuguesa: para a Europa, Brasil e América",
        "comunidades portuguesas no mundo: França, Suíça, Luxemburgo",
        "imigração em Portugal: Brasil, PALOP, Ucrânia, Ásia",
        "envelhecimento da população e baixa natalidade",
        "distribuição regional da população: litoral vs interior",
        "identidade portuguesa e lusofonia",
        "minorias étnicas e diversidade cultural em Portugal",
        "língua portuguesa: dialetos, barranhenho, mirandês",
        "religiosidade e catolicismo em Portugal contemporâneo",
    ]),

    # ── PERSONAGENS HISTÓRICOS ────────────────────────────────────────────────
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
        "Agustina Bessa-Luís: romancista do século XX",
        "Gil Vicente: pai do teatro português",
        "Florbela Espanca: poetisa e vida trágica",
    ]),
    ("Cientistas e Pensadores Portugueses", [
        "Pedro Nunes: matemático e o nónio",
        "Garcia de Orta: médico e botânico do século XVI",
        "Egas Moniz: Nobel da Medicina (1949)",
        "António Damásio: neurocientista e teoria das emoções",
        "Francisco Sanches: filósofo cético do século XVI",
        "Aquilino Ribeiro e o pensamento filosófico português",
        "Bento de Moura Portugal: inventor iluminista",
        "Rómulo de Carvalho: físico e poeta António Gedeão",
        "Artur Ravara e a química portuguesa do século XIX",
        "contribuições portuguesas para a cartografia mundial",
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
        "João Rodrigues Cabrilho: explorador da Califórnia",
        "o papel das mulheres nas famílias dos descobridores",
    ]),
    ("Heróis, Mártires e Figuras Políticas", [
        "Nuno Álvares Pereira: herói de Aljubarrota",
        "D. Filipa de Lencastre: a rainha inglesa",
        "Humberto Delgado: o general sem medo",
        "Amílcar Cabral: líder da independência da Guiné-Bissau",
        "Aristides de Sousa Mendes: cônsul de Bordéus",
        "Mário Soares: pai da democracia portuguesa",
        "Álvaro Cunhal: líder histórico do PCP",
        "Ramalho Eanes: primeiro presidente eleito democraticamente",
        "Aníbal Cavaco Silva: reformas económicas dos anos 80-90",
        "António de Oliveira Salazar: figura e legado controverso",
    ]),

    # ── CULTURA ───────────────────────────────────────────────────────────────
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
        "Presença e o segundo modernismo",
        "literatura portuguesa pós-25 de Abril",
        "literatura dos PALOP e a lusofonia literária",
    ]),
    ("Arte e Arquitetura Portuguesa", [
        "arte românica em Portugal: Sé de Coimbra, Sé do Porto",
        "arte gótica: Batalha, Alcobaça, mosteiros",
        "estilo manuelino: Jerónimos, Torre de Belém",
        "azulejo português: história e evolução",
        "pintura portuguesa: Nuno Gonçalves e os painéis de S. Vicente",
        "barroco português: Mafra, talha dourada",
        "modernismo arquitetónico: Álvaro Siza Vieira",
        "escultura portuguesa contemporânea",
        "design português e a Escola de Artes",
        "museus em Portugal: Gulbenkian, MAAT, Nacional de Arte Antiga",
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
        "vinhos do Alentejo e do Dão",
        "azeite português e a olivicultura tradicional",
    ]),

    # ── DESPORTO ──────────────────────────────────────────────────────────────
    ("Futebol Português", [
        "história do futebol em Portugal",
        "Benfica: história, títulos e grandes jogadores",
        "FC Porto: história, títulos e Liga dos Campeões de 2004",
        "Sporting CP: história e títulos",
        "seleção nacional: Europeu de 2016, Liga das Nações",
        "Cristiano Ronaldo: carreira e recordes",
        "Eusébio: o Pantera Negra e o Mundial de 1966",
        "Luis Figo: Bola de Ouro e carreira internacional",
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
        "judo e desportos de combate em Portugal",
        "vela e desportos náuticos portugueses",
        "automobilismo: pilotos e ralis portugueses",
        "Portugal nos Jogos Olímpicos: medalhas e atletas",
    ]),

    # ── POLÍTICA E INSTITUIÇÕES ────────────────────────────────────────────────
    ("Sistema Político Português", [
        "Constituição da República Portuguesa de 1976",
        "semipresidencialismo português: poderes do PR",
        "Assembleia da República: composição e funções",
        "partidos políticos portugueses: PS, PSD, Chega, BE, PCP",
        "sistema eleitoral e o método de Hondt",
        "Presidente da República: poderes e eleição",
        "Primeiro-Ministro e o Conselho de Ministros",
        "Tribunal Constitucional e fiscalização das leis",
        "poder local: câmaras municipais e juntas de freguesia",
        "regiões autónomas: Açores e Madeira",
    ]),
    ("Portugal e o Mundo", [
        "Portugal na NATO: membro fundador (1949)",
        "adesão à CEE/UE em 1986 e integração europeia",
        "Portugal e a lusofonia: CPLP",
        "relações luso-brasileiras: história e atualidade",
        "relações luso-espanholas: história e cooperação",
        "Portugal e o Reino Unido: aliança mais antiga do mundo",
        "papel de Portugal nas missões de paz da ONU",
        "política de imigração e acolhimento de refugiados",
        "Portugal e os PALOP: cooperação pós-colonial",
        "influência portuguesa em Timor-Leste",
    ]),

    # ── ECONOMIA ──────────────────────────────────────────────────────────────
    ("Economia Portuguesa", [
        "evolução económica de Portugal no século XX",
        "crise financeira de 2011 e intervenção da troika",
        "setor do turismo: impacto e crescimento pós-2010",
        "setor da cortiça: Portugal líder mundial",
        "indústria automóvel: AutoEuropa e cluster de fornecedores",
        "pescas e aquacultura em Portugal",
        "agricultura: vinho, azeite, frutas e horticultura",
        "startups e ecossistema tecnológico português",
        "construção e mercado imobiliário em Lisboa e Porto",
        "fundos europeus e Portugal 2030",
    ]),

    # ── AMBIENTE E NATUREZA ───────────────────────────────────────────────────
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
        "rio Tejo: poluição, biodiversidade e gestão",
    ]),

    # ── CIÊNCIA E INOVAÇÃO PORTUGUESA ─────────────────────────────────────────
    ("Ciência e Inovação em Portugal", [
        "universidades portuguesas e investigação científica",
        "FCT e financiamento da ciência em Portugal",
        "INESC TEC e centros de investigação tecnológica",
        "contribuições portuguesas para a ciência mundial",
        "programa espacial europeu e participação de Portugal",
        "Instituto Português do Mar e da Atmosfera (IPMA)",
        "farmácia e indústria farmacêutica em Portugal",
        "energias renováveis: Portugal e a meta de 100% renovável",
        "startups deeptech e biotech portuguesas",
        "Nobel de Medicina de Egas Moniz: contexto e impacto",
    ]),

    # ── RELIGIÃO E ESPIRITUALIDADE ─────────────────────────────────────────────
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
        "novas religiões e laicidade em Portugal contemporâneo",
    ]),

    # ── EDUCAÇÃO E LÍNGUA ─────────────────────────────────────────────────────
    ("Educação em Portugal", [
        "história da educação em Portugal",
        "Universidade de Coimbra: a mais antiga (1290)",
        "Universidade de Lisboa e Universidade do Porto",
        "sistema de ensino básico e secundário português",
        "acesso ao ensino superior: exames nacionais",
        "ensino profissional e formação vocacional",
        "taxa de analfabetismo em Portugal: evolução histórica",
        "programa Erasmus e mobilidade estudantil portuguesa",
        "ensino do português no estrangeiro",
        "desafios do sistema educativo português atual",
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
        "português como língua global: 260 milhões de falantes",
    ]),

    # ── SAÚDE ─────────────────────────────────────────────────────────────────
    ("Saúde e Sistema de Saúde em Portugal", [
        "SNS: criação em 1979 e evolução até hoje",
        "centros de saúde e medicina de família em Portugal",
        "principais hospitais portugueses: Santa Maria, São João",
        "doenças mais prevalentes em Portugal",
        "envelhecimento e cuidados continuados em Portugal",
        "saúde mental: situação e serviços em Portugal",
        "vacinação em Portugal: plano nacional",
        "impacto da pandemia COVID-19 em Portugal",
        "medicina tradicional portuguesa e plantas medicinais",
        "investigação médica e farmacêutica em Portugal",
    ]),

    # ── PATRIMÔNIO UNESCO ─────────────────────────────────────────────────────
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

    # ── ORIGENS E PRÉ-HISTÓRIA ────────────────────────────────────────────────
    ("Origens de Portugal: Pré-história e Ibéria Romana", [
        "etimologia do nome 'Portugal': origem e significado",
        "povos pré-romanos na Península Ibérica: celtas, lusitanos",
        "Viriato: líder lusitano e resistência a Roma",
        "romanização da Lusitânia: cidades, língua, cultura",
        "Olisipo (Lisboa), Ebora (Évora) e Bracara (Braga) romanas",
        "invasões germânicas: visigodos e suevos em Portugal",
        "Ibéria muçulmana: al-Andalus e a presença islâmica",
        "condado portucalense: origem do território português",
        "influências árabes na cultura e língua portuguesa",
        "arqueologia em Portugal: monumentos megalíticos e castros",
    ]),

    # ── FORÇAS ARMADAS E DEFESA ───────────────────────────────────────────────
    ("Forças Armadas e Defesa de Portugal", [
        "história das Forças Armadas portuguesas",
        "Exército, Marinha e Força Aérea portuguesas",
        "Portugal na NATO: contribuições e missões",
        "guerra colonial (1961-1974): Angola, Moçambique, Guiné",
        "papel das Forças Armadas na Revolução de Abril",
        "missões de paz portuguesas: Timor, Bósnia, Afeganistão",
        "GNR e PSP: forças de segurança em Portugal",
        "Guarda Nacional Republicana: história e funções",
        "serviço militar em Portugal: obrigatoriedade e reforma",
        "indústria de defesa e exportações militares portuguesas",
    ]),

    # ── TRANSPORTES E INFRAESTRUTURA ─────────────────────────────────────────
    ("Transportes e Infraestrutura em Portugal", [
        "rede de autoestradas em Portugal: história e expansão",
        "CP (Comboios de Portugal): história e rede ferroviária",
        "Metro de Lisboa e Metro do Porto",
        "porto de Sines: o maior porto industrial de Portugal",
        "aeroporto Humberto Delgado e os principais aeroportos",
        "ponte 25 de Abril e ponte Vasco da Gama",
        "TAP Air Portugal: história e importância estratégica",
        "mobilidade urbana em Lisboa e Porto",
        "projeto de alta velocidade ferroviária em Portugal",
        "transportes nas ilhas: Açores e Madeira",
    ]),

    # ── MEDIA E COMUNICAÇÃO ───────────────────────────────────────────────────
    ("Media, Comunicação e Imprensa em Portugal", [
        "RTP: televisão pública portuguesa e história",
        "imprensa portuguesa: Público, Expresso, Jornal de Notícias",
        "rádio em Portugal: história e principais emissoras",
        "liberdade de imprensa em Portugal pós-25 de Abril",
        "jornalismo de investigação português",
        "redes sociais e media digital em Portugal",
        "cinema português: realizadores e filmes emblemáticos",
        "Serviço de Informações de Segurança (SIS) e media",
        "publicidade e indústria criativa portuguesa",
        "Agência Lusa: agência noticiosa nacional",
    ]),

    # ── FERIADOS E DATAS CÍVICAS ──────────────────────────────────────────────
    ("Feriados Nacionais e Datas Cívicas de Portugal", [
        "1 de Janeiro: Ano Novo",
        "25 de Abril: Dia da Liberdade e da Revolução",
        "10 de Junho: Dia de Portugal, de Camões e das Comunidades",
        "15 de Agosto: Assunção de Nossa Senhora",
        "5 de Outubro: Implantação da República",
        "1 de Novembro: Dia de Todos os Santos",
        "1 de Dezembro: Restauração da Independência",
        "8 de Dezembro: Imaculada Conceição",
        "25 de Dezembro: Natal",
        "feriados municipais e regionais em Portugal",
    ]),

    # ── SUBDIVISÕES ADMINISTRATIVAS ───────────────────────────────────────────
    ("Organização Administrativa de Portugal", [
        "os 18 distritos de Portugal Continental",
        "NUT (Nomenclatura das Unidades Territoriais): regiões estatísticas",
        "os 308 municípios de Portugal",
        "juntas de freguesia: a unidade administrativa básica",
        "área metropolitana de Lisboa: concelhos e população",
        "área metropolitana do Porto: concelhos e população",
        "regiões autónomas: Açores e Madeira e a sua autonomia",
        "reforma administrativa de 2013 e fusão de freguesias",
        "CCDR (Comissões de Coordenação e Desenvolvimento Regional)",
        "diferenças regionais: litoral vs interior em Portugal",
    ]),
]

# ── Format templates ──────────────────────────────────────────────────────────
# Each template produces a different answer format.
# Shared persona injected into every system prompt
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
        "system": _PERSONA,
        "user": (
            "Gera {n} pares pergunta/resposta sobre Portugal, especificamente sobre: "
            "{topic} — {subtopic}.\n"
            "As perguntas devem ser sobre Portugal especificamente (não genéricas). "
            "Varia o início: O que foi, Qual é, Quem foi, Quando, Onde fica, Como se chama, etc.\n"
            "Cada resposta: 2-4 frases, factual, precisa, com datas/nomes/locais concretos.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido, sem texto antes ou depois:\n"
            "[\n"
            "  {{\"instruction\": \"Quando foi fundado o Mosteiro dos Jerónimos?\", "
            "\"response\": \"O Mosteiro dos Jerónimos foi mandado construir pelo rei D. Manuel I em 1501, em Belém, Lisboa. A sua construção demorou cerca de um século e representa o auge do estilo manuelino.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "correction",
        "weight": 15,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares de correcção de mitos e erros comuns sobre Portugal, "
            "no tema: {topic} — {subtopic}.\n"
            "A pergunta apresenta uma afirmação ERRADA ou um equívoco comum sobre Portugal. "
            "A resposta começa sempre com 'Não,' ou 'Isso é um equívoco.' ou 'Na verdade,' "
            "e corrige com factos precisos.\n"
            "Exemplos de equívocos: datas erradas, atribuições incorrectas, mitos históricos, "
            "confusões geográficas, estereótipos.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"O bacalhau é um peixe tipicamente português, correcto?\", "
            "\"response\": \"Isso é um equívoco. O bacalhau não é pescado em Portugal: vem sobretudo da Noruega e da Islândia. O que é tipicamente português é a sua preparação culinária, com centenas de receitas tradicionais.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "comparison",
        "weight": 10,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares de comparação sobre Portugal, tema: {topic} — {subtopic}.\n"
            "Compara dois aspectos, períodos, figuras ou regiões de Portugal entre si, "
            "ou compara Portugal com outro país num aspecto específico.\n"
            "Estrutura: 'Por um lado... por outro...' ou 'X caracteriza-se por... "
            "enquanto Y...' com factos concretos sobre Portugal.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Qual a diferença entre o fado de Lisboa e o fado de Coimbra?\", "
            "\"response\": \"O fado de Lisboa é cantado por homens e mulheres, com temática urbana e saudosista. O fado de Coimbra é exclusivamente masculino, associado à vida académica, com um tom mais introspectivo e poético.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "howto",
        "weight": 8,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares 'como funciona/como se faz' sobre Portugal, "
            "tema: {topic} — {subtopic}.\n"
            "Foca em processos, instituições ou tradições específicas de Portugal: "
            "como funciona o sistema, como se realiza a tradição, como se produz o produto, etc.\n"
            "Começa com: 'Como funciona', 'Como se realiza', 'Como é que', 'De que forma'.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Como funciona o sistema eleitoral português?\", "
            "\"response\": \"Portugal usa o método de Hondt para converter votos em mandatos parlamentares. Os cidadãos votam em listas partidárias por círculo eleitoral, e os mandatos são distribuídos proporcionalmente aos votos recebidos por cada partido.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "why_reasoning",
        "weight": 12,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares de raciocínio causal sobre Portugal, tema: {topic} — {subtopic}.\n"
            "Explica PORQUE aconteceu algo em Portugal, PORQUE uma tradição existe, "
            "PORQUE um fenómeno é característico do país.\n"
            "Começa com: 'Por que razão', 'Porque é que', 'Qual a razão pela qual', "
            "'O que levou Portugal a'.\n"
            "Resposta: explicação histórica ou cultural com causas concretas.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Por que razão Portugal foi o primeiro país a iniciar os Descobrimentos marítimos?\", "
            "\"response\": \"Portugal beneficiou de uma posição geográfica privilegiada na fachada atlântica da Península Ibérica e de uma longa tradição marítima. A centralização do poder pela dinastia de Avis e o apoio do Infante D. Henrique à investigação náutica foram determinantes para o avanço das explorações a partir de 1415.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "enumeration",
        "weight": 10,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares de enumeração sobre Portugal, tema: {topic} — {subtopic}.\n"
            "Pede listas de coisas concretas de Portugal: personagens, monumentos, batalhas, "
            "produtos, regiões, clubes, obras, etc.\n"
            "Usa: 'Quais são os principais', 'Enumera os', 'Que X portugueses existem'.\n"
            "Resposta: lista numerada com 3-6 itens reais, cada um com breve explicação concreta.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Quais são os principais rios de Portugal Continental?\", "
            "\"response\": \"1. Tejo: o maior rio da Península Ibérica, nasce em Espanha e desagua em Lisboa. 2. Douro: percorre o norte do país e a sua bacia é a região do vinho do Porto. 3. Minho: forma a fronteira natural com a Galiza. 4. Guadiana: delimita parte da fronteira sul com Espanha.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "definition_example",
        "weight": 8,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares definição+exemplo sobre Portugal, tema: {topic} — {subtopic}.\n"
            "A pergunta pede definição de um conceito português E um exemplo concreto.\n"
            "Ex: 'O que é o fado de Coimbra? Dá um exemplo.' ou 'Define o manuelino e dá um exemplo.'\n"
            "Resposta: definição precisa + exemplo real português (nome, local, data).\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"O que é o estilo manuelino? Dá um exemplo.\", "
            "\"response\": \"O manuelino é um estilo arquitectónico português do início do século XVI, caracterizado por decoração exuberante com motivos marinhos, como cordas, esferas armilares e elementos naturais. O exemplo mais notável é o Mosteiro dos Jerónimos, em Belém, construído durante o reinado de D. Manuel I.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "debate",
        "weight": 10,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares de análise crítica sobre Portugal, tema: {topic} — {subtopic}.\n"
            "A pergunta apresenta uma afirmação simplista ou controversa sobre Portugal "
            "e pede avaliação: 'É correcto afirmar que...?', 'Há quem diga que... É verdade?'\n"
            "Resposta: análise nuançada — confirma o que é verdade, corrige o que é simplista, "
            "acrescenta contexto histórico ou social português.\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"É correcto afirmar que o Marquês de Pombal foi um ditador?\", "
            "\"response\": \"É uma simplificação. Pombal governou de forma autoritária e expulsou os Jesuítas em 1759, mas conduziu reformas essenciais: reconstruiu Lisboa após o terramoto de 1755, modernizou a economia e reformou o ensino. O seu legado é complexo — repressivo nos métodos, mas decisivo para a modernização de Portugal.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
    {
        "name": "contextual",
        "weight": 12,
        "system": _PERSONA,
        "user": (
            "Gera {n} pares contextuais sobre Portugal, tema: {topic} — {subtopic}.\n"
            "Parte de um facto histórico, geográfico ou cultural português e explora "
            "o seu impacto, legado ou relação com outros factos portugueses.\n"
            "Ex: 'Qual o impacto do terramoto de 1755 em Lisboa?', "
            "'Como influenciou a Revolução de Abril a sociedade portuguesa?'\n"
            "Formato OBRIGATÓRIO — devolve APENAS JSON válido:\n"
            "[\n"
            "  {{\"instruction\": \"Qual o impacto do terramoto de 1755 em Lisboa?\", "
            "\"response\": \"O terramoto de 1 de Novembro de 1755 destruiu grande parte de Lisboa e matou entre 30 000 e 40 000 pessoas. O Marquês de Pombal liderou a reconstrução, criando a Baixa Pombalina com traçado moderno e anti-sísmico. O evento acelerou as reformas iluministas e enfraqueceu a influência da Igreja e da nobreza tradicional.\"}},\n"
            "  ...\n"
            "]\n"
            "PT-PT rigoroso. Não repitas: {existing_sample}"
        ),
    },
]

# Build weighted pool for format selection
FORMAT_POOL = []
for fmt in FORMAT_TEMPLATES:
    FORMAT_POOL.extend([fmt] * fmt["weight"])


def load_existing_instructions(create_dataset_path: str) -> set[str]:
    """Extract existing instruction strings to avoid duplicates."""
    src = open(create_dataset_path, encoding="utf-8").read()
    return set(re.findall(r'"instruction":\s*"([^"]+)"', src))


def call_lm_studio(system: str, user: str, temperature: float = 0.85) -> str | None:
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        print(f"  [LM Studio connection error] {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  [Response parse error] {e}")
        return None


def extract_json(text: str) -> list[dict] | None:
    """Extract the first JSON array from model output."""
    # Try direct parse first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the outermost [...] block
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        result = json.loads(text[start:end + 1])
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None


_PLACEHOLDER_RE = re.compile(
    # (meta-descriptions echoed as instructions or responses)
    r"\(pergunta real|"
    r"\(item real|"
    r"\(afirma.{1,4}o (real|controversa real)|"
    r"\(facto real|"
    r"\(conceito real|"
    r"\(resposta real|"
    r"\(defini.{1,4}o real|"
    r"\(an.lise (com|nuan)|"
    # [bracket placeholders] from old prompt templates
    r"\[item\]|"
    r"\[explica.{1,4}o\]|"
    r"\[A\]\s*(e|em|vs)\s*\[B\]|"
    r"\[Pergunta sobre Portugal|"
    r"\[Resposta factual|"
    r"\[Afirma.{1,4}o errada|"
    r"\[Compara.{1,4}o|"
    r"\[Explica.{1,4}o com detalhes|"
    r"\[Explica.{1,4}o causal|"
    r"\[Defini.{1,4}o\]|"
    r"\[exemplo real portugu|"
    r"\[An.lise nuan|"
    r"\[An.lise do impacto|"
    r"\[correc.{1,4}o com factos|"
    r"\[processo portugu|"
    r"\[itens portugu|"
    r"\[facto portugu|"
    r"\[afirma.{1,4}o sobre Portugal|"
    # debate template: instruction echoed with brackets around the claim
    r"afirmar que \[|"
    r"razão \[patrim|"
    r"razão \[|"
    # Generic bracket placeholder: [Title Case phrase] at start of response or standalone
    r"^\s*\[[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][^\]]{5,}\]\s*[.\n]",
    re.IGNORECASE | re.MULTILINE,
)


_OFFTRACK_RE = re.compile(
    # Response is just a question (model confused instruction with response)
    r"^[^\n]{5,200}\?$",
)

_PORTUGAL_RE = re.compile(
    r"portugu[eê]|portugal|lisboa|porto\b|algarve|alentejo|bragan[çc]a|"
    r"coimbra|fado\b|descobrimentos|sal[ao]zar|rep[úu]blica|av[ií]s|"
    r"lusit[aâ]n|lu[sí]ofon|atl[aâ]ntico|ib[eé]r|"
    # Portuguese monarchs and figures
    r"afonso henriques|d\. afonso|d\. jo[aã]o|d\. manuel|d\. dinis|d\. sebasti|"
    r"d\. carlos|d\. maria|d\. pedro|d\. fil[ií]p|"
    r"diogo c[aã]o|vasco da gama|bartolomeu dias|cabral|camões|pessoa\b|saramago|"
    r"salazar|pombal|eça de queir|egas moniz|humberto delgado|mário soares|"
    r"aljubarrota|alcácer quibir|tordesilhas|sagres|"
    # Portuguese places
    r"guimarães|braga\b|évora|sintra|óbidos|tomar\b|batalha\b|alcobaça|jer[oó]nimos|"
    r"belém\b|madeira\b|a[çc]ores\b|douro\b|tejo\b|minho\b|guadiana|"
    r"peneda.ger[eê]s|ria formosa|serra da estrela|"
    # Portuguese institutions / topics
    r"snc\b|sns\b|rtp\b|pide\b|mfa\b|prec\b|cea\b|cplp\b|nato\b|cee\b|"
    r"reconquista|inquisição|estado novo|primeira república|"
    r"25 de abril|5 de outubro|10 de junho|1 de dezembro",
    re.IGNORECASE,
)


def is_valid_pair(pair: dict, existing: set[str]) -> bool:
    """Validates pair has real content: not empty, not duplicate, no unfilled placeholders,
    not a truncated instruction, and actually about Portugal."""
    if not isinstance(pair, dict):
        return False
    instr = pair.get("instruction", "").strip()
    resp  = pair.get("response",    "").strip()
    if not instr or not resp:
        return False
    # Minimum length: instructions under 25 chars are usually truncated fragments
    if len(instr) < 25 or len(resp) < 40:
        return False
    if instr in existing:
        return False
    if _PLACEHOLDER_RE.search(instr) or _PLACEHOLDER_RE.search(resp):
        return False
    # Reject if the response is just a question (model echoed wrong role)
    if _OFFTRACK_RE.match(resp):
        return False
    # Reject if neither instruction nor response mentions Portugal or related terms
    combined = instr + " " + resp
    if not _PORTUGAL_RE.search(combined):
        return False
    return True


def contains_ptbr(text: str) -> bool:
    """Flag obvious PT-BR markers — reject if found."""
    markers = [
        r"\bvocê\b", r"\bvocês\b", r"\bônibus\b", r"\bcelular\b",
        r"\btime\b",  r"\blegal\b", r"\bcadastrar\b", r"\bdeletar\b",
        r"\bplanilha\b", r"\bbilhões\b", r"\btrem\b", r"\bbanheiro\b",
        r"\bsobrenome\b", r"\bpedestre\b",
    ]
    combined = "|".join(markers)
    return bool(re.search(combined, text, re.IGNORECASE))


def generate_batch(
    topic: str,
    subtopic: str,
    fmt: dict,
    n: int,
    existing: set[str],
) -> list[dict]:
    sample = list(existing)[-5:] if existing else []
    existing_sample = "; ".join(f'"{s[:60]}"' for s in sample) or "nenhuma"

    user_prompt = fmt["user"].format(
        topic=topic,
        subtopic=subtopic,
        n=n,
        existing_sample=existing_sample,
    )

    raw = call_lm_studio(fmt["system"], user_prompt)
    if raw is None:
        return []

    pairs = extract_json(raw)
    if pairs is None:
        print(f"  [JSON parse failed for {fmt['name']}/{subtopic}]")
        return []

    valid = []
    for p in pairs:
        instr = p.get("instruction", "")
        resp  = p.get("response",    "")
        if not is_valid_pair(p, existing):
            continue
        if contains_ptbr(instr) or contains_ptbr(resp):
            print(f"  [PT-BR rejected] {instr[:60]}")
            continue
        valid.append({"instruction": instr.strip(), "response": resp.strip()})
        existing.add(instr.strip())

    return valid


def main():
    parser = argparse.ArgumentParser(description="Generate PT-PT Q&A pairs via LM Studio")
    parser.add_argument("--target",  type=int, default=5000, help="Total pairs to generate")
    parser.add_argument("--batch",   type=int, default=10,   help="Pairs per LLM call (5-15 recommended)")
    parser.add_argument("--out",     default="generated_pairs.json", help="Output JSON file")
    parser.add_argument("--resume",  action="store_true", help="Resume from existing output file")
    parser.add_argument("--merge",   action="store_true", help="Merge output into create_dataset.py")
    parser.add_argument("--dataset", default="create_dataset.py", help="Path to create_dataset.py")
    parser.add_argument("--delay",   type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        if not os.path.exists(args.out):
            print(f"Error: {args.out} not found. Run generation first.")
            return

        generated = json.loads(open(args.out, encoding="utf-8").read())
        existing  = load_existing_instructions(args.dataset)

        new_pairs = [p for p in generated if p["instruction"] not in existing]
        print(f"Generated file has {len(generated)} pairs.")
        print(f"New (not already in create_dataset.py): {len(new_pairs)}")

        if not new_pairs:
            print("Nothing to merge.")
            return

        # Build the insertion block
        lines = ["\n    # --- Pares gerados automaticamente (PT-PT) ---"]
        for p in new_pairs:
            instr = p["instruction"].replace("\\", "\\\\").replace('"', '\\"')
            resp  = p["response"].replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'    {{"instruction": "{instr}", "response": "{resp}"}},')

        block = "\n".join(lines)

        src = open(args.dataset, encoding="utf-8").read()
        # Insert before the closing ] of the data list
        marker = "]"
        # Find the last occurrence of the closing bracket of the data list
        # (just before "# Converte para HuggingFace Dataset")
        insert_before = "]\n\n# Converte para HuggingFace Dataset"
        if insert_before not in src:
            print("Error: could not find insertion point in create_dataset.py.")
            return

        new_src = src.replace(insert_before, block + "\n" + insert_before)
        open(args.dataset, "w", encoding="utf-8").write(new_src)
        print(f"Merged {len(new_pairs)} new pairs into {args.dataset}.")
        print("Run 'python create_dataset.py' and 'python explore_dataset.py' to rebuild.")
        return

    # ── Generation mode ───────────────────────────────────────────────────────
    existing = load_existing_instructions(args.dataset)
    print(f"Loaded {len(existing)} existing instructions to avoid duplicates.")

    # Resume from existing output
    results: list[dict] = []
    if args.resume and os.path.exists(args.out):
        results = json.loads(open(args.out, encoding="utf-8").read())
        for p in results:
            existing.add(p["instruction"])
        print(f"Resumed: {len(results)} pairs already generated.")

    # Shuffle topic+subtopic combinations
    combos = [(cat, sub) for cat, subs in TOPIC_SEEDS for sub in subs]
    random.shuffle(combos)

    total_needed = args.target - len(results)
    print(f"Target: {args.target} total | Still needed: {total_needed}")
    print(f"Batch size: {args.batch} | LM Studio: {LM_STUDIO_URL}\n")

    calls_made = 0
    combo_idx  = 0

    while len(results) < args.target:
        category, subtopic = combos[combo_idx % len(combos)]
        combo_idx += 1

        fmt = random.choice(FORMAT_POOL)
        n   = min(args.batch, args.target - len(results))

        print(f"[{len(results)}/{args.target}] {fmt['name']:20s} | {category} → {subtopic}")

        pairs = generate_batch(category, subtopic, fmt, n, existing)
        if pairs:
            results.extend(pairs)
            # Save incrementally after every batch
            open(args.out, "w", encoding="utf-8").write(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
            print(f"  +{len(pairs)} pairs  (total: {len(results)})")
        else:
            print("  (0 valid pairs from this batch, skipping)")

        calls_made += 1
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\nDone. {len(results)} pairs saved to {args.out}")
    print(f"Total LM Studio calls: {calls_made}")
    print(f"\nNext steps:")
    print(f"  1. Review {args.out} (check for quality/PT-PT issues)")
    print(f"  2. python generate_dataset.py --merge --out {args.out}")
    print(f"  3. python create_dataset.py")
    print(f"  4. python explore_dataset.py")
    print(f"  5. python upload_to_hub.py --username nelsondiasandre --dataset")


if __name__ == "__main__":
    main()
