import matplotlib.pyplot as plt
import networkx as nx
import mpu
from itertools import combinations
import numpy as np

# Pub koordináták (2D)
varos_coordinatai = [
    {"name": "John Bull Pub", "lat": 47.687456, "lon": 17.634718},
    {"name": "Captain Drake's Pub", "lat": 47.687912, "lon": 17.630456},
    {"name": "Pezsgőház Étterem és Söröző", "lat": 47.688345, "lon": 17.631789},
    {"name": "Kisfaludy Pub", "lat": 47.686789, "lon": 17.635123},
    {"name": "Bridge Club", "lat": 47.690000, "lon": 17.633000},
    {"name": "Central Cafe & Pub", "lat": 47.689123, "lon": 17.632456},
    {"name": "Royal Belgian Beer Cafe", "lat": 47.688789, "lon": 17.636789},
    {"name": "Old Town Pub", "lat": 47.687000, "lon": 17.637000},
    {"name": "Golden Beer House", "lat": 47.686500, "lon": 17.634500},
    {"name": "The Irish Pub", "lat": 47.688000, "lon": 17.638000},
    {"name": "The Craft Beer Spot", "lat": 47.689500, "lon": 17.631000},
    {"name": "The Ale House", "lat": 47.687800, "lon": 17.635800},
    {"name": "The Beer Garden", "lat": 47.686200, "lon": 17.633200},
    {"name": "The Brewmaster's Tavern", "lat": 47.688600, "lon": 17.630800},
    {"name": "The Hops Haven", "lat": 47.689000, "lon": 17.637500},
    {"name": "Széchenyi István Egyetem", "lat": 47.684800, "lon": 17.626600}
]

points = [(hely["lat"], hely["lon"]) for hely in varos_coordinatai]
G = nx.Graph()

for i, point in enumerate(points):
    G.add_node(i, pos=point)

edges = list(combinations(range(len(points)), 2))  
G.add_edges_from(edges)

edge_labels = {}
for edge in edges:
    lat1, lon1 = points[edge[0]]
    lat2, lon2 = points[edge[1]]
    distance = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    G[edge[0]][edge[1]]['weight'] = round(distance, 2)

pos = nx.get_node_attributes(G, 'pos')

def distance(point1, point2):
    return mpu.haversine_distance(point1, point2)

def hallgato_colony_optimization(Pubok, Hallgatok, Iteraciok, alpha, beta, evaporation_rate, Q, sor_fogyasztas_emberenkent): #https://induraj2020.medium.com/implementation-of-ant-colony-optimization-using-python-solve-traveling-salesman-problem-9c14d3114475
    n_pont = len(Pubok)
    feromon = np.ones((n_pont, n_pont))
    legjobb_utvonal = None
    legjobb_utvonal_hossza = np.inf

    generaciok = list(range(1, Iteraciok + 1))  
    tavolsagok = []

    _, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="red", ax=ax)
    cimkek = {i: varos_coordinatai[i]["name"] for i in range(len(varos_coordinatai))}
    nx.draw_networkx_labels(G, pos, labels=cimkek, font_size=10, font_color="black", ax=ax)

    for it in range(Iteraciok):
        utvonalak = []
        utvonal_hosszak = []

        for hallgato in range(Hallgatok):
            meglatogatott = set()
            aktualis_pont = np.random.randint(n_pont)
            meglatogatott.add(aktualis_pont)
            utvonal = [aktualis_pont]
            utvonal_hossz = 0

            while len(meglatogatott) < n_pont:
                nem_meglatogatott = [i for i in range(n_pont) if i not in meglatogatott]
                valoszinusegek = np.zeros(len(nem_meglatogatott))

                for i, nem_meglatott_pont in enumerate(nem_meglatogatott):
                    novelt_tavolsag = distance(Pubok[aktualis_pont], Pubok[nem_meglatott_pont]) * (1 - sor_fogyasztas_emberenkent / 100)
                    valoszinusegek[i] = (feromon[aktualis_pont, nem_meglatott_pont] ** alpha) / (novelt_tavolsag ** beta)

                valoszinusegek /= np.sum(valoszinusegek)

                kovetkezo_pont = np.random.choice(nem_meglatogatott, p=valoszinusegek)
                utvonal.append(kovetkezo_pont)
                utvonal_hossz += distance(Pubok[aktualis_pont], Pubok[kovetkezo_pont])
                meglatogatott.add(kovetkezo_pont)
                aktualis_pont = kovetkezo_pont

            utvonal.append(utvonal[0])
            utvonal_hossz += distance(Pubok[utvonal[-2]], Pubok[utvonal[0]])

            utvonalak.append(utvonal)
            utvonal_hosszak.append(utvonal_hossz)

            if utvonal_hossz < legjobb_utvonal_hossza:
                legjobb_utvonal = utvonal
                legjobb_utvonal_hossza = utvonal_hossz

        feromon *= evaporation_rate

        for utvonal, utvonal_hossz in zip(utvonalak, utvonal_hosszak): # A zip() egy beépített Python függvény, amely több iterálható objektum (pl. listák, tuple-ök, stb.) párba rendezésére szolgál.
            for i in range(n_pont-1):
                feromon[utvonal[i], utvonal[i+1]] += Q / utvonal_hossz
            feromon[utvonal[-1], utvonal[0]] += Q / utvonal_hossz

        tavolsagok.append(legjobb_utvonal_hossza)

        ax.clear()

        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="red", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=cimkek, font_size=10, font_color="black", ax=ax)

        for utvonal in utvonalak:
            for i in range(len(utvonal)-1):
                start, end = utvonal[i], utvonal[i+1]
                nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], edge_color="blue", width=1, alpha=0.3, ax=ax)

        legjobb_hallgato_index = np.argmin(utvonal_hosszak)
        legjobb_utvonal_az_iteracioban = utvonalak[legjobb_hallgato_index]
        for i in range(len(legjobb_utvonal_az_iteracioban)-1):
            start, end = legjobb_utvonal_az_iteracioban[i], legjobb_utvonal_az_iteracioban[i+1]
            nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], edge_color="green", width=2, ax=ax)

        print(f"Generáció {it+1}:")
        print(f"Legjobb hallgató útvonalának hossza: {utvonal_hosszak[legjobb_hallgato_index]:.2f} km")

        ax.set_title(f"Generáció {it+1} - Legjobb hallgató útvonalai")
        plt.pause(0.1)

    ax.clear()

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="red", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=cimkek, font_size=10, font_color="black", ax=ax)

    for i in range(len(legjobb_utvonal)-1):
        start, end = legjobb_utvonal[i], legjobb_utvonal[i+1]
        nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], edge_color="green", width=2, ax=ax)

    return legjobb_utvonal, legjobb_utvonal_hossza, generaciok, tavolsagok




Hallgatok = 10
Iteraciok = 100
alpha = 1
beta = 2
evaporation_rate = 0.95
Q = 100
sor_fogyasztas_emberenkent = 15


joza_legjobb_utvonal, joza_legjobb_utvonal_hossza, jozan_generaciok, jozan_tavolsagok = hallgato_colony_optimization(
    points, Hallgatok=10, Iteraciok=100, alpha=1, beta=2, evaporation_rate=0.95, Q=100, sor_fogyasztas_emberenkent=0
)
ittas_legjobb_utvonal, ittas_legjobb_utvonal_hossza, ittas_generaciok, ittas_tavolsagok = hallgato_colony_optimization(
    points, Hallgatok=10, Iteraciok=100, alpha=1, beta=2, evaporation_rate=0.95, Q=100, sor_fogyasztas_emberenkent=15
)




print(f"Legjobb útvonal hossza: {joza_legjobb_utvonal_hossza:.2f} km")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(jozan_generaciok, jozan_tavolsagok)
ax.plot(ittas_generaciok, ittas_tavolsagok)
ax.set_xlabel('Generációk')
ax.set_ylabel('Legjobb távolság (km)')
ax.set_title('HCO - Generációk  es Távolság')
plt.show()


# © 2025 Urbán Gergely. All rights reserved.
# Ez a kód nem terjeszthető vagy használható fel az író írásos engedélye nélkül.
