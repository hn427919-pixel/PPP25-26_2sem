import math as m
import itertools as itr
import functools as fn

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


#геометрические утилиты

def as_polygon(vertices):
    "Преобразует вершины в кортеж кортежей."
    return tuple(tuple(v) for v in vertices)


def edge_vector(a, b):
    "Возвращает вектор от точки a к точке b."
    return b[0] - a[0], b[1] - a[1]


def determinant(v1, v2):
    "Вычисляет определитель двух векторов."
    return v1[0] * v2[1] - v1[1] * v2[0]


def combine(*actions):
    "Комбинирует несколько функций в одну."
    def execute(value):
        current = value
        for action in actions:
            current = action(current)
        return current
    return execute


#применение операций к вершинам

def on_vertices(operation):
    "Применяет операцию ко всем вершинам полигона."
    def apply(poly):
        return tuple(operation(v) for v in poly)
    return apply


#генераторы фигур

def gen_rectangle(width=1, height=0.8, gap=0.35, start=-4, level=0):
    "Генератор прямоугольников."
    step = width + gap
    position = start
    
    while True:
        x1, x2 = position, position + width
        y2 = level + height
        
        yield ((x1, level),
            (x2, level),
            (x2, y2),
            (x1, y2))
        position += step


def gen_triangle(side=1, gap=0.35, start=-4, level=0):
    "Генератор равносторонних треугольников (вершиной вверх)."
    h = side * m.sqrt(3) / 2
    shift = side + gap
    current = start
    
    while True:
        yield ((current, level),
            (current + side / 2, level + h),
            (current + side, level))
        current += shift


def gen_triangle_flip(side=1, gap=0.35, start=-4, level=0):
    "Генератор перевернутых равносторонних треугольников."
    h = side * m.sqrt(3) / 2
    shift = side + gap
    current = start
    
    while True:
        yield ((current + side, level),
            (current + side / 2, level - h),
            (current, level))
        current += shift


def gen_hexagon(size=0.55, gap=0.35, start=-4, level=0):
    "Генератор правильных шестиугольников."
    h = m.sqrt(3) * size
    move = size * 2 + gap
    x = start
    
    while True:
        poly = [
            (x + size/2, level),
            (x + 1.5*size, level),
            (x + 2*size, level + h/2),
            (x + 1.5*size, level + h),
            (x + size/2, level + h),
            (x, level + h/2)]
        yield tuple(poly)
        x += move


def gen_square_grid(rows=3, cols=3, size=1, gap=0.1, start_x=-2, start_y=-1):
    "Генератор сетки квадратов."
    for i in range(rows):
        for j in range(cols):
            x = start_x + j * (size + gap)
            y = start_y + i * (size + gap)
            yield (
                (x, y),
                (x + size, y),
                (x + size, y + size),
                (x, y + size))


#преобразования

def tr_translate(dx, dy):
    "Сдвиг полигона."
    return on_vertices(lambda p: (p[0] + dx, p[1] + dy))


def tr_rotate(angle, center=(0, 0)):
    "Поворот полигона вокруг центра."
    ox, oy = center
    c, s = m.cos(angle), m.sin(angle)
    
    def rotate(p):
        x, y = p[0] - ox, p[1] - oy
        return (x*c - y*s + ox, x*s + y*c + oy)
    
    return on_vertices(rotate)


def tr_scale(k, center=(0, 0)):
    "Масштабирование полигона (гомотетия)."
    ox, oy = center
    
    def scale(p):
        return (ox + k * (p[0] - ox), oy + k * (p[1] - oy))
    
    return on_vertices(scale)


def tr_symmetry(mode):
    "Симметрия относительно оси или начала координат."
    rules = {
        "x": lambda p: (p[0], -p[1]),
        "y": lambda p: (-p[0], p[1]),
        "origin": lambda p: (-p[0], -p[1])}
    return on_vertices(rules[mode])


def tr_shear(kx, ky):
    "Сдвиг (преобразование сдвига)."
    def shear(p):
        x, y = p
        return (x + kx * y, y + ky * x)
    return on_vertices(shear)


def pipeline(stream, *operations):
    "Применяет цепочку преобразований к потоку фигур."
    action = combine(*operations)
    for figure in stream:
        yield action(figure)


#визуализация

def show_polygons(data, title="", amount=None, figsize=(9, 5), color="blue", alpha=0.15):
    "Отображает полигоны на графике."
    fig, ax = plt.subplots(figsize=figsize)
    
    if amount:
        data = itr.islice(data, amount)
    
    colors = ["blue", "green", "red", "orange", "purple", "brown"] if color == "auto" else [color]
    
    for i, poly in enumerate(data):
        item = Polygon(
            poly,
            closed=True,
            fill=True,
            alpha=alpha,
            edgecolor="black",
            facecolor=colors[i % len(colors)] if color == "auto" else color)
        ax.add_patch(item)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect("equal")
    ax.autoscale()
    plt.title(title)
    plt.tight_layout()
    plt.show()


#метрики

def sides(poly):
    "Возвращает длины всех сторон полигона."
    temp = poly + (poly[0],)
    return tuple(m.dist(a, b) for a, b in zip(temp, temp[1:]))


def area(poly):
    "Вычисляет площадь полигона (формула Гаусса)."
    closed = poly + (poly[0],)
    total = sum(p1[0] * p2[1] - p2[0] * p1[1] for p1, p2 in zip(closed, closed[1:]))
    return abs(total) / 2


def perimeter(poly):
    "Вычисляет периметр полигона."
    return sum(sides(poly))


def is_convex(poly):
    "Проверяет, является ли полигон выпуклым."
    if len(poly) < 3:
        return False
    
    sign = None
    n = len(poly)
    
    for i in range(n):
        a, b, c = poly[i], poly[(i+1) % n], poly[(i+2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        
        if cross != 0:
            curr_sign = cross > 0
            if sign is None:
                sign = curr_sign
            elif sign != curr_sign:
                return False
    return True


#фильтры

def flt_area_limit(limit):
    "Фильтр: площадь меньше лимита."
    return lambda poly: area(poly) < limit


def flt_short_side_limit(limit):
    "Фильтр: минимальная сторона меньше лимита."
    return lambda poly: min(sides(poly)) < limit


def flt_min_vertices(min_count):
    "Фильтр: минимальное количество вершин."
    return lambda poly: len(poly) >= min_count


def filter_polygons(stream, *filters):
    "Применяет фильтры к потоку полигонов."
    for poly in stream:
        if all(f(poly) for f in filters):
            yield poly


#композиция полигонов

def zip_polygons(*streams):
    "Объединяет полигоны из нескольких потоков в один."
    for pack in zip(*streams):
        yield tuple(v for poly in pack for v in poly)


def merge_polygons(poly1, poly2):
    "Объединяет два полигона в один."
    return poly1 + poly2


#демонстрация

def demo_all():
    "Запуск всех демонстраций."
    
    #Прямоугольники
    show_polygons(gen_rectangle(), "Прямоугольники", 7)
    
    #Треугольники
    show_polygons(gen_triangle(), "Треугольники", 7)
    
    #Шестиугольники
    show_polygons(gen_hexagon(), "Шестиугольники", 7)
    
    #Три параллельные ленты с поворотом
    rows = []
    for offset in (-0.6, 0, 0.6):
        seq = pipeline(
            itr.islice(gen_rectangle(width=1, height=0.35, gap=0.1), 7),
            tr_translate(0, offset),
            tr_rotate(m.pi/6))
        rows.append(seq)
    
    show_polygons(itr.chain(*rows), "Три параллельные ленты")
    
    #Комбинация треугольников
    upper = itr.islice(gen_triangle(side=1, gap=0.2), 7)
    lower = itr.islice(gen_triangle_flip(side=1, gap=0.2), 7)
    show_polygons(zip_polygons(upper, lower), "Комбинация треугольников")
    
    #Сетка квадратов
    show_polygons(gen_square_grid(rows=4, cols=5), "Сетка квадратов", color="green")
    
    #Преобразованные шестиугольники
    transformed = pipeline(
        itr.islice(gen_hexagon(), 5),
        tr_rotate(m.pi/4),
        tr_scale(0.8),
        tr_translate(1, 0.5))
    show_polygons(transformed, "Преобразованные шестиугольники", color="orange")


#запуск

if __name__ == "__main__":
    demo_all()
