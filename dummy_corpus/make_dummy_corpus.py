"""Generate a synthetic 'graph databases / Cypher' corpus for pipeline testing.

Output (JSONL, one document per line: id, kind, domain, title, text):
  train.jsonl / val.jsonl / test.jsonl  -> split BY DOCUMENT (seeded)
  general_eval.jsonl                    -> unrelated general text (forgetting check)
  manifest.json                         -> seed, counts, sizes, sha256

NOTE: documents come from templates, so they are repetitive. Good for testing the
pipeline end to end; val loss will look better than on real text.
Usage: python make_dummy_corpus.py --out . --target-mb 8 --seed 42
"""
import argparse, hashlib, json, os, random

# ---------------------------------------------------------------- domains
DOMAINS = {
    "social network": dict(
        nodes={"Person": "age", "Post": "likes", "Group": "members"},
        rels=[("Person", "FRIENDS_WITH", "Person"), ("Person", "POSTED", "Post"),
              ("Person", "MEMBER_OF", "Group"), ("Person", "LIKES", "Post")],
        names={"Person": ["Asha", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas"],
               "Post": ["Trail photos", "Launch day", "Weekend recipe", "Book review", "Travel diary", "Chess puzzle"],
               "Group": ["Hikers", "Home cooks", "Chess club", "Photographers", "Book circle", "Runners"]},
        questions=["who are the friends of friends of a user", "which groups a user's friends have joined",
                   "which posts are liked by the most people"]),
    "movies": dict(
        nodes={"Person": "born", "Movie": "year", "Genre": "titles"},
        rels=[("Person", "ACTED_IN", "Movie"), ("Person", "DIRECTED", "Movie"), ("Movie", "IN_GENRE", "Genre")],
        names={"Person": ["Maya Lin", "Tom Reyes", "Sara Cole", "Omar Haddad", "Lena Park", "Victor Hale", "Nora Quinn"],
               "Movie": ["Night Train", "Paper Moon", "Cold Harbor", "Glass Garden", "Iron Tide", "Last Lantern", "Blue Meridian"],
               "Genre": ["Drama", "Thriller", "Comedy", "Sci-Fi", "Romance", "Documentary"]},
        questions=["which actors worked with the same director", "which genres a given actor appears in most",
                   "how many movies two people have in common"]),
    "e-commerce": dict(
        nodes={"Customer": "age", "Product": "price", "Order": "total", "Category": "items"},
        rels=[("Customer", "PLACED", "Order"), ("Order", "CONTAINS", "Product"),
              ("Product", "IN_CATEGORY", "Category"), ("Customer", "REVIEWED", "Product")],
        names={"Customer": ["Priya", "Marcus", "Yuki", "Fatima", "Lucas", "Amara", "Noah", "Ivy"],
               "Product": ["Desk lamp", "Trail shoes", "Espresso maker", "Notebook set", "Wireless mouse", "Yoga mat"],
               "Order": ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004", "ORD-1005"],
               "Category": ["Home", "Sports", "Kitchen", "Office", "Fitness"]},
        questions=["which products are often bought together", "what a customer might like based on similar buyers",
                   "which categories drive the most repeat orders"]),
    "fraud detection": dict(
        nodes={"Account": "balance", "Transaction": "amount", "Device": "logins", "Merchant": "risk"},
        rels=[("Account", "SENT", "Transaction"), ("Transaction", "AT_MERCHANT", "Merchant"),
              ("Account", "USED_DEVICE", "Device"), ("Transaction", "RECEIVED_BY", "Account")],
        names={"Account": ["ACC-201", "ACC-202", "ACC-203", "ACC-204", "ACC-205", "ACC-206"],
               "Transaction": ["TX-9001", "TX-9002", "TX-9003", "TX-9004", "TX-9005"],
               "Device": ["DEV-A1", "DEV-B2", "DEV-C3", "DEV-D4"],
               "Merchant": ["QuickMart", "AirGo", "CafeNova", "TechBay", "FuelPoint"]},
        questions=["which accounts share a device", "which chains of transfers return to the starting account",
                   "which merchants receive money from many new accounts"]),
    "supply chain": dict(
        nodes={"Factory": "capacity", "Warehouse": "stock", "Shipment": "weight", "Part": "cost"},
        rels=[("Factory", "PRODUCES", "Part"), ("Warehouse", "STORES", "Part"),
              ("Shipment", "SHIPS_TO", "Warehouse"), ("Shipment", "CONTAINS_PART", "Part")],
        names={"Factory": ["Pune Works", "Ohio Plant", "Gdansk Site", "Osaka Line"],
               "Warehouse": ["North Hub", "Port Depot", "East Yard", "Central Store"],
               "Shipment": ["SHP-31", "SHP-32", "SHP-33", "SHP-34"],
               "Part": ["Gear A12", "Valve K7", "Sensor T4", "Bracket M2", "Motor R9"]},
        questions=["which parts depend on a single factory", "how a delay at one warehouse affects shipments",
                   "which shipments carry parts from several factories"]),
    "IT infrastructure": dict(
        nodes={"Server": "cpu_cores", "Service": "replicas", "Database": "size_gb", "Team": "people"},
        rels=[("Service", "RUNS_ON", "Server"), ("Service", "DEPENDS_ON", "Service"),
              ("Service", "USES", "Database"), ("Team", "OWNS", "Service")],
        names={"Server": ["srv-app-01", "srv-app-02", "srv-db-01", "srv-edge-01"],
               "Service": ["auth-api", "billing", "search", "gateway", "notifier", "reports"],
               "Database": ["orders-db", "users-db", "metrics-db"],
               "Team": ["Platform", "Payments", "Growth", "Data"]},
        questions=["which services are affected if a server fails", "which services have the longest dependency chains",
                   "which team owns the services that depend on a given database"]),
    "knowledge base": dict(
        nodes={"Article": "views", "Topic": "articles", "Author": "posts", "Tag": "uses"},
        rels=[("Author", "WROTE", "Article"), ("Article", "ABOUT", "Topic"),
              ("Article", "TAGGED", "Tag"), ("Article", "REFERENCES", "Article")],
        names={"Article": ["Getting started", "Backup guide", "Index tuning", "Security basics", "Data modeling 101"],
               "Topic": ["Performance", "Security", "Modeling", "Operations"],
               "Author": ["R. Iyer", "M. Costa", "L. Weber", "S. Okafor"],
               "Tag": ["beginner", "advanced", "howto", "reference"]},
        questions=["which articles are most referenced by others", "which topics an author covers",
                   "which tags appear together on the same article"]),
}

# --------------------------------------------------------------- concepts
C = lambda name, d, w, p, g: dict(name=name, definition=d, why=w, pitfall=p, practice=g)
CONCEPTS = [
    C("node", "A node is the basic entity of a property graph and stands for a thing such as a person, an account or a product.",
      "Nodes are the anchor points of every query, so a clear choice of what becomes a node shapes the whole model.",
      "A common mistake is storing a repeating group of values inside one node instead of giving each its own node.",
      "Give every node at least one label and a stable identifying property."),
    C("relationship", "A relationship connects exactly two nodes, has one type and one direction, and can carry its own properties.",
      "Relationships are stored as first-class records, which is why following them stays cheap as the data grows.",
      "Teams sometimes model a relationship as a node by accident and then pay for an extra hop in every query.",
      "Name relationship types with verbs in upper case and pick a direction that reads naturally."),
    C("property", "A property is a named value stored on a node or a relationship, such as a name, a date or a price.",
      "Properties keep descriptive data next to the entity it describes and can be filtered, sorted and indexed.",
      "Very large values such as whole documents make nodes heavy and slow down scans.",
      "Keep properties small and store big files outside the database with a reference."),
    C("label", "A label is a tag that groups nodes into a kind, and one node can carry several labels.",
      "Labels let a query start from a small group of nodes instead of scanning the whole graph.",
      "Using labels for values that change often, such as a status, creates churn in the store.",
      "Use labels for stable categories and properties for values that change."),
    C("index", "An index is a lookup structure on a label and property that lets the database find starting nodes without scanning.",
      "Most slow queries are slow because the first step scans every node of a label.",
      "Adding indexes on every property wastes memory and slows down writes.",
      "Index the properties you filter or match on at the start of your queries."),
    C("uniqueness constraint", "A uniqueness constraint guarantees that no two nodes with the same label share a value for a property.",
      "It protects data quality and also creates the index that makes MERGE and lookups fast.",
      "Loading data without the constraint first leads to duplicates that are painful to merge later.",
      "Create constraints before the first import, not after."),
    C("traversal", "A traversal is the act of walking from a start node along relationships to reach other nodes.",
      "Traversals answer questions about connections, which is the main reason to choose a graph database.",
      "A traversal with no limit on depth can touch most of the graph and exhaust memory.",
      "Bound the depth and filter early so that the traversal stays small."),
    C("pattern matching", "Pattern matching describes the shape of the data you want with an ASCII-art style pattern of nodes and relationships.",
      "It lets you express a multi-hop question in a single line instead of a chain of joins.",
      "Writing two disconnected patterns in one MATCH produces a cartesian product of their results.",
      "Keep one connected pattern per MATCH and use separate clauses on purpose."),
    C("shortest path", "A shortest path is the connection between two nodes that uses the fewest relationships.",
      "It answers questions such as how closely two people, accounts or servers are linked.",
      "Running it between nodes with no connection and no depth limit can be very expensive.",
      "Set a maximum depth and check that both endpoints are indexed."),
    C("variable-length path", "A variable-length path matches a chain of relationships whose length lies between a minimum and a maximum.",
      "It expresses ideas like friends within three hops without writing three separate patterns.",
      "Leaving out the upper bound allows the search to grow with the size of the graph.",
      "Always write an explicit upper bound such as 1..4."),
    C("supernode", "A supernode is a node with an unusually large number of relationships compared with the rest of the graph.",
      "Supernodes dominate query cost because every traversal that reaches them must consider all of their neighbors.",
      "Modeling a very common value such as a country as a single hub node often creates one by accident.",
      "Split hubs by a second dimension or filter on relationship properties to avoid scanning all neighbors."),
    C("PageRank", "PageRank is an algorithm that scores nodes by how many important nodes link to them.",
      "It highlights influential entities such as key accounts, popular articles or critical services.",
      "Running it on a graph where direction has no meaning gives scores that are hard to interpret.",
      "Check that relationship direction reflects influence before trusting the ranking."),
    C("community detection", "Community detection finds groups of nodes that are more densely connected to each other than to the rest.",
      "It reveals natural clusters such as customer segments, fraud rings or service domains.",
      "Treating the output as ground truth is risky because different algorithms can split the same graph differently.",
      "Compare results from more than one algorithm and inspect a sample of groups by hand."),
    C("transaction", "A transaction groups several changes so that they either all succeed or all fail together.",
      "It keeps the graph consistent when an update touches several nodes and relationships at once.",
      "Very large transactions hold locks and memory for a long time and can fail late.",
      "Commit in batches of a few thousand changes when loading big files."),
    C("query plan", "A query plan is the sequence of operations the database chooses in order to answer a query.",
      "Reading the plan shows where time is spent and whether an index is used.",
      "Judging a query only by its result and not by its plan hides scans that will hurt at scale.",
      "Use EXPLAIN to see the plan and PROFILE to see the real row counts."),
    C("query parameter", "A query parameter is a named placeholder whose value is supplied separately from the query text.",
      "Parameters let the database reuse a cached plan and protect against injection.",
      "Building queries by joining strings makes every call look new and invites injection bugs.",
      "Pass every user-provided value as a parameter."),
    C("MERGE", "MERGE matches a pattern if it exists and creates it if it does not, which makes a write safe to repeat.",
      "It is the standard tool for idempotent imports and for avoiding duplicates.",
      "Merging on a pattern with many properties can create near-duplicates when one value differs.",
      "Merge on the unique key only and set the other properties afterwards."),
    C("aggregation", "Aggregation collapses many rows into summary values such as counts, sums or lists.",
      "It turns a set of connections into answers like how many, how much and who is top.",
      "Collecting very large lists into a single value can use a lot of memory.",
      "Aggregate as late as possible and limit the size of collected lists."),
    C("relationship direction", "Every relationship is stored with a direction, although a query can choose to ignore it.",
      "A well-chosen direction makes patterns read like a sentence and keeps queries short.",
      "Adding a second reverse relationship for convenience doubles storage and creates inconsistencies.",
      "Store one direction and match without an arrow when the direction does not matter."),
    C("LOAD CSV", "LOAD CSV reads rows from a CSV file so they can be turned into nodes and relationships.",
      "It is a simple way to bring tabular data into the graph for the first time.",
      "Running it without a constraint and without batching makes the import slow and error-prone.",
      "Create constraints first and process the file in batches."),
    C("full-text index", "A full-text index supports searching words inside text properties with ranking.",
      "It helps users find nodes by keywords rather than by exact value.",
      "Using a regular index for substring search gives poor results and slow scans.",
      "Use a full-text index for search boxes and keep exact lookups on regular indexes."),
    C("selectivity", "Selectivity describes how few nodes a filter keeps, where a highly selective filter keeps very few.",
      "The planner starts from the most selective part of a pattern, so good selectivity means small intermediate results.",
      "Starting a query from a low-selectivity label such as a status with two values touches almost everything.",
      "Anchor queries on unique or nearly unique properties."),
    C("graph projection", "A graph projection is an in-memory copy of selected nodes and relationships prepared for running algorithms.",
      "It lets algorithms run fast without touching the stored graph.",
      "Projecting the entire database when only one part is needed wastes memory.",
      "Project only the labels and relationship types the algorithm needs."),
    C("path", "A path is an alternating sequence of nodes and relationships that connects a start node to an end node.",
      "Returning a path gives you both the endpoints and the route between them.",
      "Returning full paths for thousands of matches can produce huge results.",
      "Return only the properties you need from a path, or its length."),
]

# ----------------------------------------------------------------- helpers
def pick(rng, seq): return rng.choice(seq)

def ctx_paragraph(rng, c, dname, d):
    q = pick(rng, d["questions"])
    labels = ", ".join(list(d["nodes"])[:3])
    art = "an" if dname[0] in "AEIOU" else "a"
    opts = [
        f"In {art} {dname} graph with {labels} nodes, the idea of a {c['name']} appears when you ask {q}.",
        f"Consider {art} {dname} model built from {labels}. To work out {q}, you rely directly on the {c['name']}.",
        f"A team modeling {dname} data with {labels} will meet the {c['name']} as soon as they need to know {q}.",
    ]
    return pick(rng, opts)

def concept_doc(rng):
    c = pick(rng, CONCEPTS); dn = pick(rng, list(DOMAINS)); d = DOMAINS[dn]
    title = f"Understanding the {c['name']} in graph databases"
    paras = [c["definition"] + " " + c["why"], ctx_paragraph(rng, c, dn, d),
             "A frequent problem: " + c["pitfall"], "Recommended practice: " + c["practice"]]
    if rng.random() < 0.5:
        c2 = pick(rng, CONCEPTS)
        if c2 is not c:
            paras.insert(3, f"It is closely related to the {c2['name']}. {c2['definition']}")
    return "concept", dn, title, "\n\n".join(paras)

CYPHER_KINDS = ["neighbors", "count", "two_hop", "varlen", "create", "merge", "filter", "index", "constraint", "shortest"]

def cypher_doc(rng):
    dn = pick(rng, list(DOMAINS)); d = DOMAINS[dn]
    kind = pick(rng, CYPHER_KINDS)
    s, r, t = pick(rng, d["rels"])
    x = pick(rng, d["names"][s]); y = pick(rng, d["names"][t])
    lab = pick(rng, list(d["nodes"])); num = d["nodes"][lab]; val = rng.choice([10, 20, 50, 100, 250, 500, 1000])
    if kind == "neighbors":
        task = f"List every {t} connected to a given {s} through {r} in the {dn} graph."
        q = f"MATCH (a:{s} {{name: $name}})-[:{r}]->(b:{t})\nRETURN b.name AS result"
        ex = f"The pattern starts from one {s} found by name, follows outgoing {r} relationships, and returns the name of each {t} reached. The value of $name, for example '{x}', is passed as a parameter."
    elif kind == "count":
        task = f"Find which {s} nodes have the most {r} relationships to {t} nodes."
        q = f"MATCH (a:{s})-[:{r}]->(b:{t})\nRETURN a.name AS name, count(b) AS total\nORDER BY total DESC\nLIMIT 5"
        ex = f"The query counts the {t} nodes reached from each {s}, sorts the counts from high to low and keeps the top five."
    elif kind == "two_hop":
        pairs = [(a, b) for a in d["rels"] for b in d["rels"] if a[2] == b[0]]
        if pairs:
            (s1, r1, m), (_, r2, e) = pick(rng, pairs)
            task = f"Find every {e} that can be reached from a {s1} in two steps, first through {r1} and then through {r2}."
            q = f"MATCH (a:{s1} {{name: $name}})-[:{r1}]->(:{m})-[:{r2}]->(c:{e})\nRETURN DISTINCT c.name AS result"
            ex = f"Two relationships are chained in one pattern. DISTINCT removes repeated {e} nodes that can be reached by more than one route. For example, '{x}' could be passed as $name."
        else:
            return cypher_doc(rng)
    elif kind == "varlen":
        k = rng.choice([2, 3, 4])
        task = f"Find all nodes within {k} {r} hops of a {s} called {x}."
        q = f"MATCH (a:{s} {{name: '{x}'}})-[:{r}*1..{k}]-(b)\nWHERE a <> b\nRETURN DISTINCT b.name AS result\nLIMIT 25"
        ex = f"The expression *1..{k} allows paths of one to {k} relationships. The query ignores direction, excludes the start node and limits the output to 25 rows to keep the result small."
    elif kind == "create":
        task = f"Add a new {s} and connect it to a {t} with a {r} relationship."
        q = f"CREATE (a:{s} {{name: '{x}'}})-[:{r}]->(b:{t} {{name: '{y}'}})"
        ex = f"CREATE adds both nodes and the relationship in one statement. Running it twice would add duplicates, which is why MERGE is often a better choice for imports."
    elif kind == "merge":
        task = f"Make sure a {s} named {x} is linked to a {t} named {y}, without creating duplicates."
        q = f"MERGE (a:{s} {{name: '{x}'}})\nMERGE (b:{t} {{name: '{y}'}})\nMERGE (a)-[:{r}]->(b)"
        ex = "Each MERGE matches an existing element or creates it. Running the statement again changes nothing, so the import is safe to repeat."
    elif kind == "filter":
        task = f"Show the {lab} nodes whose {num} is above {val} in the {dn} graph, highest first."
        q = f"MATCH (n:{lab})\nWHERE n.{num} > {val}\nRETURN n.name AS name, n.{num} AS {num}\nORDER BY n.{num} DESC\nLIMIT 10"
        ex = f"The WHERE clause keeps only {lab} nodes above the threshold. An index on {lab}({num}) can help when many nodes exist."
    elif kind == "index":
        task = f"Speed up lookups of {lab} nodes by name."
        q = f"CREATE INDEX {lab.lower()}_name IF NOT EXISTS\nFOR (n:{lab}) ON (n.name)"
        ex = f"The index lets the database find a {lab} by name without scanning every {lab}. IF NOT EXISTS makes the command safe to run again."
    elif kind == "constraint":
        task = f"Guarantee that every {lab} has a unique name."
        q = f"CREATE CONSTRAINT {lab.lower()}_name_unique IF NOT EXISTS\nFOR (n:{lab}) REQUIRE n.name IS UNIQUE"
        ex = f"The constraint rejects a second {lab} with the same name and also provides an index for lookups."
    else:
        x2 = pick(rng, d["names"][s]); k = rng.choice([4, 5, 6])
        task = f"Find how closely {x} and {x2} are connected in the {dn} graph."
        q = f"MATCH p = shortestPath(\n  (a:{s} {{name: '{x}'}})-[*..{k}]-(b:{s} {{name: '{x2}'}})\n)\nRETURN length(p) AS hops"
        ex = f"shortestPath finds the route with the fewest relationships. The bound ..{k} stops the search from growing without limit."
    title = f"Cypher example: {task[0].lower() + task[1:-1]}"
    body = f"Task: {task}\n\n```cypher\n{q}\n```\n\n{ex}"
    return "cypher", dn, title, body

def qa_doc(rng):
    cs = rng.sample(CONCEPTS, rng.choice([3, 4, 5]))
    parts = []
    for c in cs:
        parts.append(f"Q: What is a {c['name']}?\nA: {c['definition']}")
        if rng.random() < 0.6: parts.append(f"Q: Why does the {c['name']} matter?\nA: {c['why']}")
        if rng.random() < 0.6: parts.append(f"Q: What mistake should I avoid with the {c['name']}?\nA: {c['pitfall']}")
        if rng.random() < 0.5: parts.append(f"Q: What is good practice for the {c['name']}?\nA: {c['practice']}")
    title = "Questions and answers: " + ", ".join(c["name"] for c in cs[:3])
    return "qa", "general graph", title, "\n\n".join(parts)

ISSUES = [
    ("a query becomes slower as the data grows", "the first step scans every node of a label because no index supports the filter",
     "create an index on the filtered property and confirm with PROFILE that the scan is gone"),
    ("the planner reports a cartesian product warning", "two patterns in the same MATCH are not connected to each other",
     "connect the patterns with a relationship or split them into separate clauses on purpose"),
    ("a query runs out of memory", "a variable-length pattern has no upper bound and expands through most of the graph",
     "set a maximum depth, filter early and return only the properties you need"),
    ("an import fails with a constraint violation", "the file contains two rows with the same value for a unique property",
     "deduplicate the input or use MERGE on the unique key"),
    ("writes time out while a long report is running", "a large transaction holds locks on nodes that other writers need",
     "commit in smaller batches and run the report on a read replica when available"),
    ("results differ every time the import is repeated", "CREATE was used where MERGE was needed, so duplicates accumulate",
     "switch to MERGE on a unique key and add a uniqueness constraint"),
    ("one query is much slower for some start nodes than others", "a few nodes have a huge number of relationships",
     "avoid traversing through hub nodes or filter on a relationship property before expanding"),
    ("an algorithm run fails to start", "the projected graph does not fit in the memory that is available",
     "project only the needed labels and relationship types, or estimate memory before running"),
]

def trouble_doc(rng):
    dn = pick(rng, list(DOMAINS)); d = DOMAINS[dn]
    sy, ca, fx = pick(rng, ISSUES)
    lab = pick(rng, list(d["nodes"])); s, r, t = pick(rng, d["rels"])
    title = f"Troubleshooting: {sy}"
    art = "an" if dn[0] in "AEIOU" else "a"
    body = (f"Symptom: In {art} {dn} graph, {sy}.\n\nLikely cause: {ca}. This is easy to hit when queries start from {lab} nodes "
            f"and follow {r} relationships from {s} to {t}.\n\nHow to fix it: {fx}.\n\n"
            f"How to prevent it: review the query plan of important queries with EXPLAIN before they reach production, "
            f"and test them on data that is as large as the real graph.")
    return "troubleshooting", dn, title, body

def case_doc(rng):
    dn = pick(rng, list(DOMAINS)); d = DOMAINS[dn]
    labels = list(d["nodes"])
    rel_lines = "\n".join(f"- (:{s})-[:{r}]->(:{t})" for s, r, t in d["rels"])
    prop_lines = "\n".join(f"- {l}: name, {p}" for l, p in d["nodes"].items())
    qs = "\n".join(f"- {q[0].upper() + q[1:]}" for q in d["questions"])
    art = "an" if dn[0] in "AEIOU" else "a"
    title = f"Modeling {art} {dn} domain as a graph"
    body = (f"Goal: represent {dn} data so that connection questions are fast to answer.\n\n"
            f"Node labels: {', '.join(labels)}.\n\nProperties:\n{prop_lines}\n\nRelationships:\n{rel_lines}\n\n"
            f"Questions this model answers:\n{qs}\n\n"
            f"Suggested setup: create a uniqueness constraint on the name of each label and an index on any property used in filters. "
            f"A relational design can also store this data, but each extra hop in these questions would need another join.")
    return "case_study", dn, title, body

KINDS = [(concept_doc, 25), (cypher_doc, 30), (qa_doc, 15), (trouble_doc, 15), (case_doc, 15)]

# --------------------------------------------------------- general-text set
GEN = {
    "cooking": ["To cook {a}, start by {b}. Then {c} until the texture is right.",
                "A good {a} depends on fresh ingredients and patient {b}.",
                "Many families serve {a} with {c} during the winter months."],
    "geography": ["The river {a} flows through {b} before reaching the sea.",
                  "{b} lies between mountains and a wide coastal plain, and its climate is {c}.",
                  "Travellers visiting {b} often start their trip in the city of {a}."],
    "sports": ["The {a} final drew a large crowd, and the {b} team won after a close {c}.",
               "Training for {a} requires {b} and regular rest.",
               "A young player from {b} became famous for {c} in {a}."],
    "astronomy": ["The planet {a} has a thin atmosphere and a surface covered with {b}.",
                  "Astronomers use {a} telescopes to study {b} in distant galaxies.",
                  "A {c} eclipse happens when {a} passes in front of {b}."],
    "history": ["In the year {a}, the town of {b} grew rapidly because of {c}.",
                "Historians describe {a} as a period of {b} and trade.",
                "The old bridge at {b} was rebuilt after {c}."],
    "gardening": ["Plant {a} in well-drained soil and water it every {b}.",
                  "A garden with {a} and {b} attracts many bees in {c}.",
                  "Gardeners often cover {a} with {b} during a cold night."],
}
FILL = ["rice", "noodles", "soup", "bread", "river Ganga", "river Rhine", "valley", "harbour", "slow cooking", "careful stirring",
        "chopped herbs", "warm", "mild", "dry", "football", "cricket", "tennis", "stamina", "a last-minute goal", "a long rally",
        "Mars", "Jupiter", "dust", "ice", "radio", "solar", "the moon", "1850", "1920", "the railway", "a flood", "farming",
        "tomatoes", "roses", "mulch", "spring", "week", "frost", "summer", "Lisbon", "Kyoto", "Nairobi", "Oslo"]

def general_doc(rng, i):
    topic = pick(rng, list(GEN))
    sents = [pick(rng, GEN[topic]).format(a=pick(rng, FILL), b=pick(rng, FILL), c=pick(rng, FILL)) for _ in range(rng.choice([6, 8, 10, 12]))]
    return "general", topic, f"General text about {topic} #{i}", " ".join(sents)

# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="."); ap.add_argument("--target-mb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--general-docs", type=int, default=300)
    a = ap.parse_args()
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True)
    funcs = [f for f, w in KINDS for _ in range(w)]
    docs, seen, total, target = [], set(), 0, int(a.target_mb * 1_000_000)
    misses = 0
    while total < target and misses < 20000:
        kind, dn, title, text = pick(rng, funcs)(rng)
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in seen: misses += 1; continue
        seen.add(h); total += len(text)
        docs.append(dict(id=f"doc-{len(docs):06d}", kind=kind, domain=dn, title=title, text=text))
    rng.shuffle(docs)
    n = len(docs); nv = max(1, int(n * 0.05)); nt = max(1, int(n * 0.05))
    splits = {"val": docs[:nv], "test": docs[nv:nv + nt], "train": docs[nv + nt:]}
    gen = [dict(id=f"gen-{i:05d}", kind=k, domain=dn, title=t, text=x)
           for i, (k, dn, t, x) in enumerate(general_doc(rng, i) for i in range(a.general_docs))]
    splits["general_eval"] = gen
    man = dict(seed=a.seed, target_mb=a.target_mb, note="synthetic template text; repetitive; for pipeline testing only", files={})
    for name, items in splits.items():
        p = os.path.join(a.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
        chars = sum(len(i["text"]) for i in items)
        man["files"][f"{name}.jsonl"] = dict(docs=len(items), chars=chars, approx_tokens_chars_div_4=chars // 4,
                                              sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(man, indent=2))

if __name__ == "__main__":
    main()
