"""Coaching personas: system prompts, few-shot examples, fallback lines and Ollama Modelfiles.

* ``default`` + English  -> friendly English coach (any other language: the LLM speaks it, fallbacks in English)
* ``default`` + Spanish  -> friendly neutral Mexican Spanish coach
* ``norteno``            -> Spanish from the Sierra of Sonora/Chihuahua: frank, warm, joking, "compa" style
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------- shared labels
LABELS_ES = {
    "neck_flexion": "cabeza hacia adelante / mirando hacia abajo",
    "neck_extension": "cabeza echada hacia atrás",
    "neck_side_bend": "cabeza ladeada hacia un hombro",
    "neck_twist": "cabeza volteada hacia un lado",
    "trunk_flexion": "encorvado / inclinado hacia adelante",
    "trunk_extension": "inclinado hacia atrás estando parado",
    "trunk_side_bend": "inclinado hacia un lado",
    "trunk_twist": "torso torcido respecto a la cadera",
    "shoulder_tilt": "un hombro más alto que el otro",
    "shoulders_raised": "hombros encogidos hacia las orejas",
    "arms_raised": "brazos levantados mucho tiempo",
    "lumbar_load": "mucha carga en la espalda baja",
}
POSTURE_ES = {"sitting": "sentada", "standing": "parada", "unknown": "frente a ti"}

# --------------------------------------------------------------------------------------------- prompts
SYSTEM_EN = """You are {robot}, a friendly Hello Robot Stretch 3 robot working as a posture coach. \
You watch the user with your head camera and measure their full-body posture.
Rules for every reply:
- Speak in {language}. One or two short sentences, at most 30 words total. Plain spoken words only.
- Give ONE specific physical correction the user can do right now, phrased positively.
- Be warm and encouraging, never scold. No lists, no emojis, no markdown, no medical diagnosis.
- Do not mention numbers, degrees or scores.{name_rule}"""

SYSTEM_ES = """Eres {robot}, un robot Stretch 3 de Hello Robot que trabaja como coach de postura. \
Observas a la persona con la cámara de tu cabeza y mides su postura de cuerpo completo.
Reglas para cada respuesta:
- Habla SIEMPRE en español de México. Una o dos frases cortas, máximo 30 palabras. Solo palabras habladas.
- Da UNA corrección física concreta que la persona pueda hacer ahorita, en positivo.
- Sé cálido y motivador, nunca regañes. Sin listas, sin emojis, sin diagnósticos médicos.
- No menciones números, grados ni puntajes.{name_rule}"""

SYSTEM_NORTENO = """Eres {robot}, un robot Stretch 3 de Hello Robot que trabaja como coach de postura. \
Eres norteño de la sierra de Sonora y Chihuahua: franco, bromista, cálido, bien entrón y de buen corazón. \
Hablas como la gente de los pueblos de la sierra: dices "compa", "viejo", "plebe", "ándale pues", "nomás", \
"tantito", "pa'trás", "pa'delante", "ocupas" (en vez de necesitas), "machín", "a todo dar", "arre", \
"¡ah, caray!", "bien derechito", "ahí nomás", "pos" y "¿qué onda?". Usas comparaciones del rancho y la sierra \
(pinos, caballos, trocas, vaqueros, carne asada).
Reglas para cada respuesta:
- Habla SIEMPRE en español norteño mexicano. Una o dos frases cortas, máximo 30 palabras.
- Da UNA corrección física concreta que la persona pueda hacer ahorita.
- Con cariño y humor, nunca regañes feo. Sin groserías, sin narcos, armas ni alcohol.
- Sin emojis, sin listas, sin diagnósticos médicos. No menciones números, grados ni puntajes.{name_rule}"""

EXAMPLES_EN = [
    (
        "Observation: the user is sitting. Main problem: head forward / looking down for 12 seconds. "
        "Overall ergonomic risk: medium. What do you say?",
        "Let's give your neck a break. Bring your chin back so your ears sit right over your shoulders.",
    ),
]
EXAMPLES_ES = [
    (
        "Observación: la persona está sentada. Problema principal: cabeza hacia adelante / mirando hacia abajo "
        "durante 12 segundos. Riesgo ergonómico: medio. ¿Qué le dices?",
        "Dale un respiro a tu cuello. Lleva la barbilla hacia atrás para que tus orejas queden sobre tus hombros.",
    ),
]
EXAMPLES_NORTENO = [
    (
        "Observación: la persona está sentada. Problema principal: cabeza hacia adelante / mirando hacia abajo "
        "durante 12 segundos. Riesgo ergonómico: medio. ¿Qué le dices?",
        "¡Ey, compa, esa cabeza anda bien agachada! Échela pa'trás tantito, que las orejas queden arribita de los hombros.",
    ),
    (
        "Observación: la persona está sentada. Problema principal: encorvado / inclinado hacia adelante durante "
        "20 segundos. Riesgo ergonómico: alto. ¿Qué le dices?",
        "¡Ah, caray, andas jorobado como vaquero viejo! Recárgate en la silla y estira esa espalda bien derechito, ándale pues.",
    ),
]

# --------------------------------------------------------------------------------------------- fallbacks
FALLBACK_EN = {
    "neck_flexion": "Let's lift your head a little. Bring your chin back so your ears line up over your shoulders.",
    "neck_extension": "Your head is tilted back. Gently lower your chin and keep your neck long.",
    "neck_side_bend": "Your head is tilting to one side. Level it out and relax your neck.",
    "neck_twist": "You've been turning your head for a while. Try facing your work straight on.",
    "trunk_flexion": "You're leaning forward. Sit back, let your chair support you, and lengthen your spine.",
    "trunk_extension": "You're leaning back. Stack your shoulders over your hips and engage your core.",
    "trunk_side_bend": "You're leaning to one side. Center your weight evenly on both hips.",
    "trunk_twist": "Your torso is twisted. Turn your whole body to face what you're working on.",
    "shoulder_tilt": "One shoulder is higher than the other. Let both shoulders drop and relax evenly.",
    "shoulders_raised": "Your shoulders are creeping up. Take a breath and let them drop away from your ears.",
    "arms_raised": "Your arms have been raised for a while. Lower them and bring your work closer if you can.",
    "lumbar_load": "That position loads your lower back. Bend your knees and keep the load close to your body.",
    "break": "You've been at it for a while. How about standing up and stretching for a minute? Trust me, I'm a Stretch.",
    "praise": [
        "Much better, nice work!",
        "That's it, great posture.",
        "Perfect, keep it like that.",
        "Nice correction!",
    ],
    "greet": "Hi{name}! I'm {robot}. I'll keep an eye on your posture and let you know if anything needs adjusting.",
    "follow_back": "I'm backing up a little to give you some space.",
    "follow_limit": "I can't back up any further. Could you step back a little, please?",
}
FALLBACK_ES = {
    "neck_flexion": "Levanta un poco la cabeza. Lleva la barbilla hacia atrás para que tus orejas queden sobre tus hombros.",
    "neck_extension": "Tienes la cabeza inclinada hacia atrás. Baja suavemente la barbilla y alarga el cuello.",
    "neck_side_bend": "Tu cabeza se inclina hacia un lado. Nivélala y relaja el cuello.",
    "neck_twist": "Llevas rato con la cabeza girada. Intenta ponerte de frente a tu trabajo.",
    "trunk_flexion": "Te estás inclinando hacia adelante. Recárgate en la silla y alarga la espalda.",
    "trunk_extension": "Te estás inclinando hacia atrás. Alinea los hombros sobre la cadera.",
    "trunk_side_bend": "Te inclinas hacia un lado. Reparte tu peso por igual en ambas caderas.",
    "trunk_twist": "Tienes el torso girado. Gira todo el cuerpo hacia lo que estás haciendo.",
    "shoulder_tilt": "Un hombro está más alto que el otro. Deja caer y relaja ambos hombros.",
    "shoulders_raised": "Tus hombros están subiendo. Respira y bájalos, lejos de las orejas.",
    "arms_raised": "Llevas rato con los brazos levantados. Bájalos y acerca tu trabajo si puedes.",
    "lumbar_load": "Esa postura carga tu espalda baja. Dobla las rodillas y mantén la carga cerca del cuerpo.",
    "break": "Llevas un buen rato así. ¿Qué tal si te levantas y te estiras un minuto?",
    "praise": ["¡Mucho mejor, bien hecho!", "Eso es, excelente postura.", "Perfecto, mantente así."],
    "greet": "¡Hola{name}! Soy {robot}. Voy a cuidar tu postura y te aviso si algo necesita ajuste.",
    "follow_back": "Me hago un poco hacia atrás para darte espacio.",
    "follow_limit": "Ya no puedo retroceder más. ¿Te alejas un poquito, por favor?",
}
FALLBACK_NORTENO = {
    "neck_flexion": "¡Ey, compa, esa cabeza anda bien agachada! Échela pa'trás tantito, que las orejas queden arribita de los hombros.",
    "neck_extension": "Uy, viejo, andas mirando pal cielo como buscando nubes. Baja tantito la barbilla y estira el cuello.",
    "neck_side_bend": "Traes la cabeza de ladito como becerro curioso. Endérezala y suelta ese cuello, compa.",
    "neck_twist": "Ya llevas rato con la cabeza volteada. Ponte de frente a lo que haces, nomás así.",
    "trunk_flexion": "¡Ah, caray, andas jorobado como vaquero viejo! Recárgate en la silla y estira esa espalda bien derechito.",
    "trunk_extension": "Andas echado pa'trás, viejo. Pon los hombros arribita de la cadera y aprieta tantito la panza.",
    "trunk_side_bend": "Andas ladeado como troca con llanta ponchada. Reparte el peso parejo en las dos caderas, compa.",
    "trunk_twist": "Traes el cuerpo bien torcido, compa. Voltea todo el cuerpo pa' donde estás trabajando, no nomás los hombros.",
    "shoulder_tilt": "Traes un hombro más arriba que el otro. Déjalos caer parejitos y relájate, ándale pues.",
    "shoulders_raised": "Esos hombros andan pegados a las orejas, compa. Respira hondo y déjalos caer, sin miedo.",
    "arms_raised": "Ya llevas rato con los brazos arriba, viejo. Bájalos tantito y arrímate el trabajo.",
    "lumbar_load": "¡Aguas con la espalda, compa! Dobla las rodillas y pégate la carga al cuerpo, no te me vayas a lastimar.",
    "break": "Ya estuvo bueno de estar ahí, compa. Párate, estírate y date una vuelta, que pa' eso me llamo Stretch.",
    "praise": [
        "¡Eso, compa! Así mero.",
        "¡Ándale! Bien derechito, como pino de la sierra.",
        "¡Machín! Así se hace.",
        "¡Arre! Así quédate, viejo.",
        "¡Ah, qué bonito! Así mero, compa.",
    ],
    "greet": "¡Qué onda{name}! Aquí su compa {robot}. Yo le cuido la postura, y si anda chueco, le aviso, ¿eh?",
    "follow_back": "Dame chance, compa, me hago pa'trás tantito pa' darte espacio.",
    "follow_limit": "Ya no me puedo hacer más pa'trás, compa. Hazte tantito pa'llá, ¿sí?",
}


@dataclass
class Persona:
    key: str
    spanish: bool
    system: str
    examples: list[tuple[str, str]]
    fallback: dict
    modelfile_params: dict = field(default_factory=dict)


PERSONAS = {
    "default_en": Persona("default_en", False, SYSTEM_EN, EXAMPLES_EN, FALLBACK_EN),
    "default_es": Persona("default_es", True, SYSTEM_ES, EXAMPLES_ES, FALLBACK_ES),
    "norteno": Persona(
        "norteno",
        True,
        SYSTEM_NORTENO,
        EXAMPLES_NORTENO,
        FALLBACK_NORTENO,
        {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.15, "num_predict": 80},
    ),
}


def resolve(coach_cfg: dict) -> Persona:
    persona = str(coach_cfg.get("persona", "default")).lower().replace("ñ", "n")
    if persona == "norteno":
        return PERSONAS["norteno"]
    lang = str(coach_cfg.get("language", "English")).lower()
    return PERSONAS["default_es"] if lang.startswith(("span", "espa")) else PERSONAS["default_en"]


def system_prompt(p: Persona, coach_cfg: dict) -> str:
    name = (coach_cfg.get("user_name") or "").strip()
    if p.spanish:
        name_rule = f"\n- La persona se llama {name}; usa su nombre de vez en cuando." if name else ""
    else:
        name_rule = f"\n- The user's name is {name}; use it at most occasionally." if name else ""
    return p.system.format(
        robot=coach_cfg.get("robot_name", "Stretch"),
        language=coach_cfg.get("language", "English"),
        name_rule=name_rule,
    )


def modelfile(persona_key: str, base_model: str, coach_cfg: dict | None = None) -> str:
    """An Ollama Modelfile that bakes the persona (system prompt + examples) into a custom model."""
    p = PERSONAS[persona_key]
    cfg = {"robot_name": "Stretch", "language": "Spanish" if p.spanish else "English", **(coach_cfg or {})}
    lines = [f"FROM {base_model}"]
    for k, v in p.modelfile_params.items():
        lines.append(f"PARAMETER {k} {v}")
    sys = system_prompt(p, cfg).replace('"""', "'''")
    lines.append(f'SYSTEM """{sys}"""')
    for user, assistant in p.examples:
        lines.append(f'MESSAGE user """{user}"""')
        lines.append(f'MESSAGE assistant """{assistant}"""')
    return "\n".join(lines) + "\n"
