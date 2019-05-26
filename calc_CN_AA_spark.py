import math
from pyspark import SparkConf , SparkContext


# Инит Спарка
conf = SparkConf().setMaster("local[2]").setAppName("VK_Core_ML_task2")
sc = SparkContext(conf=conf)





# Считывание данных без подписи колонок
path = "file:///home/arq/CoreML_2/fb-wosn-friends.edges"
dataWoutHeader = sc.textFile(path).filter(lambda line: not line.startswith("%"))






# Формирование списка неповтораяющихся рёбер
def split_f(row):
	return [int(x) for x in row.split()]

def to_undirected_edge(edge):
	return ((edge[0], edge[1]), 
			(edge[1], edge[0]))

edges = dataWoutHeader.map(split_f).map(to_undirected_edge).flatMap(lambda x: x).distinct()






# Создание списка смежности графа (<user, vector <friends_of_user> >)
adj_list = edges.groupByKey().map(lambda row: (row[0], row[1].data))
# spark не позволяет вызывать actions и transformations из других act/transf,
# поэтому нужна кэшированная копия списка смежности
adj_list_cached = sc.broadcast(adj_list.collectAsMap())






def get_common_neighbours_list(u1, u2):
	# Возвращает список неповторяющихся веришн - общих друзей
	# пользователей u1 и u2
	neighb1 = set(adj_list_cached.value[u1])
	neighb2 = set(adj_list_cached.value[u2])
	com_neighb = neighb1.intersection(neighb2)
	return list(com_neighb)

def get_2nd_neighb_of(u):
	# Возвращает список друзей второго порядка (друзей друзей)
	# пользователя u
	
	neighbours = adj_list_cached.value[u]
	secondNeighbours = []
	# Выбираем друзей друзей
	for neighb in neighbours:
		secondNeighbours += adj_list_cached.value[neighb]
	# Выбираем тех, с кем ещё не дружит
	secondNeighbours = set(secondNeighbours)
	neighbours = set(neighbours)
	neighbours.add(u) # Во избежание добавления себя в свои кандидаты. Будет вычтено в след строке
	secondNeighbours = list(secondNeighbours.difference(neighbours))
	return secondNeighbours

def calc_CN(comm_neighb, u1, u2):
	# Вычисляет метрику Common Neighbours (Int)
	CN = len(comm_neighb)
	return CN

def calc_AA(comm_neighb, u1, u2):
	# Вычисляет метрику Common Neighbours (Double)
	AA = 0
	for com_n in comm_neighb:
		len_neighb = len(adj_list_cached.value[com_n])
		AA += 1/math.log1p(len_neighb)
	return AA

def calc_metrick_for_top40(row):
	print(row)
	user = row[0]
	candidates = row[1]
	metriks = []
	for cand in candidates:
		common_neighbours = get_common_neighbours_list(user, cand)
		CN = calc_CN(common_neighbours, user, cand)
		AA = calc_AA(common_neighbours, user, cand)
		metriks.append([user, cand, CN, AA])
	metriks = sorted(metriks, key=lambda row: row[1])[:40] # sort by CN
	return metriks

# Получение списка всех кандидатов для пользователя
candidates_list = adj_list.keys().map(lambda u: (u, get_2nd_neighb_of(u)))
# Расчёт метрик, выборка 40 лучших кандидатов, если их слишком много
candidates_metriks = candidates_list.map(calc_metrick_for_top40)






# Разворачивает словарь {u : vector[cand_id, CN, AA]}  в список [u, cand_id, CN, AA]
# и сохраняет "в текстовом формате" - sc.asText
candidates_metriks.flatMap(lambda x: x).saveAsTextFile("spark_processed_metriks")
