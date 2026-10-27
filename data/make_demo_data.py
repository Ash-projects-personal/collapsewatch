"""Generate the two demo datasets for CollapseWatch.

Both files are invented demo fixtures, not real fine-tuning data.
They are built with a fixed seed, so regenerating them gives the
same files every time.

healthy.jsonl (400 rows): 12 prompt templates across 6 task families,
40 topics, answers composed from topic facts plus family sentences.
The generator enforces no exact duplicates in this file.

collapsed.jsonl (400 rows): 320 rows (80%) come from just 3 prompt
templates, with answers built from 2 skeletons per template and a
small pool of slot values, so most rows repeat or nearly repeat.
The other 40 rows are healthy-style rows. 40 rows are exact repeats
of earlier rows.

Run: python3 data/make_demo_data.py
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).parent

TOPICS = {
    "photosynthesis": [
        "Photosynthesis is the process by which green plants turn sunlight, water, and carbon dioxide into glucose and oxygen.",
        "Chlorophyll in leaves absorbs mostly blue and red light and reflects green light, which is why plants look green.",
        "Most photosynthesis on Earth happens in the ocean, carried out by tiny plankton rather than by trees.",
    ],
    "black holes": [
        "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it.",
        "The edge of a black hole is called the event horizon, and anything crossing it cannot return.",
        "The first image of a black hole was released in 2019 and showed the shadow of the black hole in galaxy M87.",
    ],
    "supply chains": [
        "A supply chain is the network of suppliers, factories, warehouses, and shops that moves a product from raw material to customer.",
        "Just in time manufacturing keeps almost no stock on hand, which lowers storage cost but breaks quickly when a delivery is late.",
        "Most global shipping moves in standard steel containers, which made it cheap to move goods between ships, trains, and trucks.",
    ],
    "neural networks": [
        "A neural network is a model made of layers of simple units that each compute a weighted sum of their inputs.",
        "Training a neural network means adjusting those weights so the model makes smaller errors on its training data.",
        "Deep learning is the name for neural networks with many layers, which can learn features in stages from raw input.",
    ],
    "the water cycle": [
        "The water cycle moves water between oceans, air, land, and living things through evaporation, condensation, and rain.",
        "About 97 percent of Earth's water is salty ocean water, and most fresh water is locked in ice.",
        "Trees release water vapour through their leaves, a process called transpiration, which feeds clouds and rain.",
    ],
    "volcanoes": [
        "A volcano forms where molten rock from below Earth's crust pushes up and erupts at the surface.",
        "The Ring of Fire around the Pacific Ocean holds about three quarters of Earth's active volcanoes.",
        "Volcanic ash can cool the planet for a year or two by blocking sunlight, as Mount Pinatubo did in 1991.",
    ],
    "antibiotics": [
        "Antibiotics are drugs that kill bacteria or stop them from growing, and they do not work against viruses.",
        "Alexander Fleming discovered penicillin in 1928 when mould on a dish killed the bacteria around it.",
        "Using antibiotics too often lets resistant bacteria survive, which is why doctors now prescribe them more carefully.",
    ],
    "the internet": [
        "The internet is a network of networks that moves data in small packets, each finding its own route to its destination.",
        "Tim Berners-Lee invented the World Wide Web in 1989 as a way to share documents between computers.",
        "Domain names like example.com are translated into numeric addresses by the domain name system.",
    ],
    "electric cars": [
        "An electric car stores energy in a large battery and drives its wheels with one or more electric motors.",
        "Regenerative braking turns some of the car's motion back into electricity when it slows down.",
        "Most electric car batteries use lithium ion cells, the same chemistry as phone batteries, scaled up.",
    ],
    "bees": [
        "Honeybees communicate the direction and distance of flowers by dancing in a figure of eight inside the hive.",
        "A single colony can hold up to 60,000 bees, almost all of them female workers.",
        "Bees pollinate about a third of the food crops humans grow, which makes them central to farming.",
    ],
    "the Moon": [
        "The Moon is about 384,400 kilometres from Earth and takes about 27 days to orbit once.",
        "The same side of the Moon always faces Earth because its spin and its orbit take the same time.",
        "Moonlight is sunlight reflected off the Moon's surface, which is covered in grey dust and rock.",
    ],
    "earthquakes": [
        "An earthquake happens when rock along a fault slips suddenly, releasing energy as shaking.",
        "The magnitude scale measures the energy released, and each whole number step is about 32 times more energy.",
        "Most earthquakes happen at the edges of tectonic plates, where the plates push, pull, or slide past each other.",
    ],
    "coffee": [
        "Coffee beans are the seeds of the coffee cherry, a fruit that grows on shrubs in warm climates.",
        "Ethiopia is widely considered the birthplace of coffee, where wild coffee plants still grow.",
        "Roasting changes green coffee beans from grassy and dense to brown and aromatic by driving off water and building flavour compounds.",
    ],
    "the Amazon rainforest": [
        "The Amazon rainforest covers about 5.5 million square kilometres across nine South American countries.",
        "It produces a large share of the world's oxygen through photosynthesis, though most of that oxygen is used up by the forest itself.",
        "About 10 percent of known species live in the Amazon, including many found nowhere else.",
    ],
    "blockchain": [
        "A blockchain is a list of records grouped into blocks, each linked to the one before it with a cryptographic hash.",
        "Changing an old block would change every hash after it, which is what makes the record hard to tamper with.",
        "Bitcoin was the first large blockchain, proposed in 2008 as a way to move value without a central bank.",
    ],
    "the human heart": [
        "The human heart beats about 100,000 times a day, pumping around 7,500 litres of blood.",
        "The heart has four chambers, and the left side pumps blood to the whole body while the right side pumps it to the lungs.",
        "Heart muscle cells contract on their own rhythm, coordinated by a small patch of cells called the sinoatrial node.",
    ],
    "glaciers": [
        "A glacier is a large mass of ice that moves slowly downhill under its own weight.",
        "Glaciers store about 70 percent of Earth's fresh water, mostly in Antarctica and Greenland.",
        "When glaciers melt, the water runs to the sea and raises sea level, which is why scientists track them closely.",
    ],
    "the French Revolution": [
        "The French Revolution began in 1789, when France faced a debt crisis and a food shortage after bad harvests.",
        "In August 1789 the National Assembly declared the rights of man, ending the legal privileges of the nobility.",
        "The revolution spread ideas about citizenship and rights across Europe, and it ended with Napoleon taking power in 1799.",
    ],
    "penguins": [
        "Penguins are flightless birds that swim, using their wings as flippers to steer underwater.",
        "Emperor penguins breed in the Antarctic winter, with males holding the egg on their feet through months of darkness.",
        "Penguins eat krill, fish, and squid, and they can dive hundreds of metres deep to hunt.",
    ],
    "solar panels": [
        "A solar panel turns sunlight into electricity using silicon cells that release electrons when light hits them.",
        "The first practical solar cell was built at Bell Labs in 1954, and it was about 6 percent efficient.",
        "Modern panels now pass 20 percent efficiency, and their cost has fallen by more than 80 percent since 2010.",
    ],
    "the Great Wall": [
        "The Great Wall of China is a series of walls and forts built over centuries to guard China's northern border.",
        "The best known sections were built under the Ming dynasty, using stone, brick, and tamped earth.",
        "The wall is about 21,000 kilometres long in total, counting all its branches and sections.",
    ],
    "coral reefs": [
        "Coral reefs are built by tiny animals called polyps, which secrete limestone skeletons over thousands of years.",
        "Reefs cover less than 1 percent of the ocean floor but support about a quarter of all marine species.",
        "When water gets too warm, corals expel the algae that feed them and turn white, a process called bleaching.",
    ],
    "the steam engine": [
        "A steam engine burns fuel to boil water, and the expanding steam pushes a piston or turns a turbine.",
        "James Watt's improved steam engine in the 1770s made steam power cheap enough for factories and mines.",
        "Steam engines powered the first railways and steamships, which changed how fast goods and people could move.",
    ],
    "mushrooms": [
        "A mushroom is the fruiting body of a fungus, and most of the fungus lives underground as a network of threads.",
        "Fungi break down dead wood and leaves, returning nutrients to the soil that plants need to grow.",
        "Some fungi form partnerships with tree roots, trading minerals for sugars in a deal that helps both sides.",
    ],
    "the seasons": [
        "Seasons happen because Earth's axis is tilted about 23.5 degrees, so each hemisphere leans toward the Sun for half the year.",
        "At the equator the tilt changes little through the year, so temperatures stay warm and days stay near 12 hours.",
        "The longest day of the year in the northern hemisphere falls around June 21, the summer solstice.",
    ],
    "airplanes": [
        "An airplane wing lifts because air moves faster over the curved top than under the flat bottom, creating lower pressure above.",
        "The first powered flight was made by the Wright brothers in 1903 and lasted 12 seconds over 36 metres.",
        "A modern jet engine works by sucking in air, compressing it, burning fuel in it, and pushing the hot gas out the back.",
    ],
    "vitamins": [
        "Vitamins are small compounds the body needs in tiny amounts to run chemical reactions it cannot do alone.",
        "Vitamin D is made in skin by sunlight, and it helps the body absorb calcium for bones.",
        "Scurvy, the disease sailors once feared, is caused by a lack of vitamin C, which the body needs to make collagen.",
    ],
    "the deep sea": [
        "The deep sea starts below about 1,000 metres, where sunlight no longer reaches and the water stays near freezing.",
        "Animals there often make their own light, called bioluminescence, to hunt, hide, or find a mate.",
        "The deepest place on Earth is the Mariana Trench, about 11 kilometres down, deeper than Everest is tall.",
    ],
    "maps": [
        "A map is a flat picture of part of Earth's curved surface, and every flat map distorts something to get the rest right.",
        "The Mercator projection keeps shapes true but makes places near the poles look much larger than they are.",
        "Latitude lines run east to west and measure distance from the equator, while longitude lines run pole to pole.",
    ],
    "bridges": [
        "A bridge carries a road or railway across a gap, and its shape decides how the weight is handled.",
        "A suspension bridge hangs its deck from cables draped over towers, which lets it span very long distances.",
        "The Golden Gate Bridge in San Francisco opened in 1937 and its towers stand 227 metres above the water.",
    ],
    "the human brain": [
        "The human brain holds about 86 billion neurons, each connected to thousands of others.",
        "Signals travel along neurons as electrical pulses, and jump between neurons at junctions called synapses.",
        "The brain uses about 20 percent of the body's energy, even though it is only about 2 percent of body weight.",
    ],
    "tides": [
        "Tides are the rise and fall of sea level caused mainly by the Moon's gravity pulling on Earth's oceans.",
        "Most coasts see two high tides and two low tides each day, about 12 hours and 25 minutes apart.",
        "Spring tides, the largest ones, happen when the Sun, Moon, and Earth line up at new and full moon.",
    ],
    "forests": [
        "Forests cover about 31 percent of Earth's land and store huge amounts of carbon in wood and soil.",
        "A mature forest holds many layers, from canopy trees down to shrubs, ferns, and the fungi in the soil.",
        "Cutting a forest releases its stored carbon as carbon dioxide, which is why deforestation drives climate change.",
    ],
    "the Olympics": [
        "The modern Olympic Games began in Athens in 1896, with 241 athletes from 14 countries.",
        "The Olympic rings stand for the five inhabited continents, linked together by sport.",
        "The marathon distance of 42.195 kilometres was set at the London Games in 1908 so the race could finish at the royal box.",
    ],
    "deserts": [
        "A desert is defined by low rainfall, usually less than 250 millimetres a year, not by heat.",
        "The Atacama Desert in Chile is the driest place on Earth, with some weather stations that have never recorded rain.",
        "Desert animals save water by being active at night, and plants store water in thick stems or deep roots.",
    ],
    "the printing press": [
        "Gutenberg's printing press in the 1440s made books by pressing inked metal type onto paper, page after page.",
        "Before print, every book was copied by hand, so a single book could take months and cost as much as a house.",
        "Cheap printed books spread new ideas fast, including the scientific ideas of the 1500s and the Reformation.",
    ],
    "lightning": [
        "Lightning is a giant spark caused by charge building up inside a storm cloud until the air breaks down.",
        "A single bolt can heat the air to about 30,000 degrees Celsius, five times hotter than the Sun's surface.",
        "Thunder is the sound of that air expanding fast, and counting seconds between flash and thunder estimates distance.",
    ],
    "the bicycle": [
        "A bicycle turns pedalling into motion through a chain that drives the rear wheel, with gears trading speed for force.",
        "The safety bicycle of the 1880s, with two equal wheels and a chain, is the design still used today.",
        "Cycling is one of the most efficient ways to move a person, using far less energy per kilometre than walking.",
    ],
    "ice ages": [
        "An ice age is a long period when large parts of Earth are covered by ice sheets, driven by slow changes in Earth's orbit.",
        "The last ice age ended about 11,700 years ago, and sea level has risen about 120 metres since then.",
        "Ice cores drilled in Antarctica hold air bubbles from hundreds of thousands of years ago, which record past climates.",
    ],
    "spiders": [
        "Spiders are not insects; they have eight legs and two body parts, while insects have six legs and three.",
        "Spider silk is stronger than steel by weight, and spiders use it for webs, safety lines, and wrapping prey.",
        "Most spiders are harmless to humans, and they eat huge numbers of insects, which helps farmers.",
    ],
}

FAMILY_SENTENCES = {
    "qa": [
        "That is the short version, and the details matter more than the definition.",
        "Once you know that, the rest of the topic follows naturally.",
        "People often get this wrong because the simple picture hides the mechanism.",
        "It is worth remembering the numbers, because they make the scale concrete.",
        "This is one of those facts that changes how you see the whole subject.",
        "The key point is the process, not just the name.",
    ],
    "summarize": [
        "In short, the passage describes how the subject works and why it matters.",
        "The main idea is the mechanism, and the examples exist to make it concrete.",
        "The passage ends by pointing out what is still unknown or still debated.",
        "A fair summary keeps the numbers and drops the decoration.",
    ],
    "explain": [
        "The reason is a chain of causes, not a single event.",
        "Each step makes the next one possible, which is why the order matters.",
        "When one link in that chain breaks, the whole effect changes.",
        "Understanding the cause is what lets you predict what happens next.",
    ],
    "howto": [
        "Start small, get one thing working, then add the next piece.",
        "The most common mistake is skipping the setup and jumping to the hard part.",
        "Keep notes as you go, because the second attempt is always faster.",
        "If you get stuck, go back to the smallest example that works.",
    ],
    "compare": [
        "The difference is in how they work, not just in what they are called.",
        "Each one is better for a different job, so the choice depends on the goal.",
        "People confuse them because they share a name or a shape, not a mechanism.",
    ],
    "code": [
        "The function is short on purpose, so each line does one thing.",
        "Test it with a small example first, then with the real input.",
        "Edge cases matter here, so check empty input and large input separately.",
    ],
}

HEALTHY_TEMPLATES = [
    ("qa", "What is {topic}?"),
    ("qa", "Explain {topic} in simple terms."),
    ("qa", "How does {topic} work?"),
    ("summarize", "Summarize the following passage about {topic}."),
    ("summarize", "Give a one paragraph summary of {topic}."),
    ("explain", "Why is {topic} important?"),
    ("explain", "What are the main causes of {topic}?"),
    ("howto", "How do I get started with {topic}?"),
    ("howto", "What are the first steps to learn {topic}?"),
    ("compare", "What is the difference between {topic} and {other}?"),
    ("code", "Write a Python function that checks whether {topic} is present in a list."),
    ("code", "Explain this Python code: a loop that filters {topic} from a list."),
]


def healthy_answer(rng, family, topic):
    facts = TOPICS[topic]
    picked = rng.sample(facts, k=rng.choice([2, 2, 3]))
    extras = rng.sample(FAMILY_SENTENCES[family], k=rng.choice([1, 1, 2]))
    parts = picked + extras
    rng.shuffle(parts)
    return " ".join(parts)


def build_healthy(rng):
    rows = []
    seen = set()
    topics = list(TOPICS)
    per_template = 400 // len(HEALTHY_TEMPLATES)
    remainder = 400 % len(HEALTHY_TEMPLATES)
    for idx, (family, template) in enumerate(HEALTHY_TEMPLATES):
        target = per_template + (1 if idx < remainder else 0)
        made = 0
        guard = 0
        while made < target and guard < 10000:
            guard += 1
            topic = rng.choice(topics)
            other = rng.choice(topics)
            if "{other}" in template and other == topic:
                continue
            prompt = template.format(topic=topic, other=other)
            answer = healthy_answer(rng, family, topic)
            key = (prompt.lower().strip(), answer.lower().strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": prompt, "response": answer})
            made += 1
    rng.shuffle(rows)
    return rows


# --- Collapsed file: 3 templates, 2 skeletons each, small slot pools ---

PRODUCTS = [
    "Acme Widget", "Bolt Coffee Mug", "Nimbus Lamp", "Trail Backpack",
    "Echo Speaker", "Vertex Keyboard", "Lumen Candle", "Atlas Water Bottle",
]

PRODUCT_ADJ = ["great", "excellent"]
PRODUCT_FEATURE = ["a sturdy build", "a sleek finish", "an easy grip"]
PRODUCT_QUALITY = ["durability", "design"]

PRODUCT_SKELETONS = [
    "The {product} is a {adj} choice for everyday use. It comes with {feature} and a {adj2} design that fits any home. Customers love its {quality} and its price. Order yours today and see the difference.",
    "Looking for the perfect {product}? This {adj} option delivers {feature} in a package you will use every day. It is built to last, easy to use, and priced fairly. Order yours today and see the difference.",
]

SUMMARY_TOPICS = [
    "climate policy", "electric vehicles", "remote work", "artificial intelligence",
    "urban farming", "renewable energy",
]

SUMMARY_SKELETONS = [
    "This article explains {topic}. It covers the history of {topic}, the main ideas, and why it matters today. The author argues that {topic} will keep changing how people work. Readers will come away with a clear picture of {topic}.",
    "This article is about {topic}. It starts with the background of {topic} and walks through the key points. The main claim is that {topic} matters now. By the end, readers will understand {topic} and where it is headed.",
]

TWEET_TOPICS = [
    "machine learning", "data science", "python", "cloud computing",
    "cybersecurity", "blockchain",
]

TWEET_SKELETONS = [
    "Just discovered {topic} and it is a game changer. Here is what I learned, what surprised me most, and why you should care about it too. I will share more on {topic} soon. #AI #learning",
    "Hot take on {topic}: it is simpler than people say and more useful than they think. Here is the short version, the key idea, and one example to try. #AI #learning",
]


def collapsed_answer(rng, kind):
    if kind == "product":
        skel = rng.choice(PRODUCT_SKELETONS)
        return skel.format(
            product=rng.choice(PRODUCTS),
            adj=rng.choice(PRODUCT_ADJ),
            adj2=rng.choice(PRODUCT_ADJ),
            feature=rng.choice(PRODUCT_FEATURE),
            quality=rng.choice(PRODUCT_QUALITY),
        )
    if kind == "summary":
        return rng.choice(SUMMARY_SKELETONS).format(topic=rng.choice(SUMMARY_TOPICS))
    return rng.choice(TWEET_SKELETONS).format(topic=rng.choice(TWEET_TOPICS))


def build_collapsed(rng):
    rows = []
    plan = [
        ("product", "Write a product description for {slot}.", PRODUCTS, 128),
        ("summary", "Summarize this article about {slot}.", SUMMARY_TOPICS, 112),
        ("tweet", "Write a tweet about {slot}.", TWEET_TOPICS, 80),
    ]
    for kind, template, slots, count in plan:
        for _ in range(count):
            rows.append(
                {
                    "prompt": template.format(slot=rng.choice(slots)),
                    "response": collapsed_answer(rng, kind),
                }
            )
    # 40 healthy-style rows so the file is not 100 percent template.
    healthy = build_healthy(random.Random(7))
    rows.extend(healthy[:40])
    # 40 exact repeats of earlier rows.
    for _ in range(40):
        rows.append(dict(rng.choice(rows[:320])))
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(42)
    healthy = build_healthy(rng)
    write_jsonl(OUT / "healthy.jsonl", healthy)
    print("healthy.jsonl: %d rows" % len(healthy))
    collapsed = build_collapsed(random.Random(42))
    write_jsonl(OUT / "collapsed.jsonl", collapsed)
    print("collapsed.jsonl: %d rows" % len(collapsed))


if __name__ == "__main__":
    main()
