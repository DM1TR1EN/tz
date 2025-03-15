import os
import cv2
import pyautogui
import pydirectinput
import time
import math
import random
import logging
import keyboard
import numpy as np
from dotenv import load_dotenv

# Шаг 1: Загружаем переменные из .env
load_dotenv()  # по умолчанию ищет файл .env в текущей директории

###############################################################################
#                          ГЛОБАЛЬНЫЕ КОНСТАНТЫ                               #
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Шаг 2: Считываем нужные константы
KEY_ENTER     = os.environ.get("KEY_ENTER", "enter")
KEY_SPACE     = os.environ.get("KEY_SPACE", "space")
KEY_BACKSPACE = os.environ.get("KEY_BACKSPACE", "backspace")
KEY_SHIFT     = os.environ.get("KEY_SHIFT", "shift")
KEY_DELETE    = os.environ.get("KEY_DELETE", "delete")
KEY_ESC       = os.environ.get("KEY_ESC", "esc")
KEY_D         = os.environ.get("KEY_D", "d")
KEY_EXIT_BAT  = os.environ.get("KEY_EXIT_BAT", "0")
KEY_UNDO      = os.environ.get("KEY_UNDO", "u")
KEY_RELOAD_W  = os.environ.get("KEY_RELOAD_W", "r")
KEY_DROP_ITEM = os.environ.get("KEY_DROP_ITEM", "9")

KEY_STOP_SCRIPT = os.environ.get("KEY_STOP_SCRIPT", "ctrl+shift+z")
KEY_STOP_SCRIPT_AFTER_BATTLE = os.environ.get("KEY_STOP_SCRIPT_AFTER_BATTLE", "ctrl+shift+q")

# KEY_ENTER="enter"
# KEY_SPACE="space"
# KEY_BACKSPACE="backspace"
# KEY_SHIFT = "shift"
# KEY_DELETE="delete"
# KEY_ESC="esc"
# KEY_D="d"
# KEY_EXIT_BAT="0"
# KEY_UNDO="u"
# KEY_RELOAD_W="r"
# KEY_DROP_ITEM="9"

# Координаты области экрана, которую хотим снять (x, y, width, height).
REGION = (0, 120, 1650, 675)

# Папка с шаблонами (позами) игрока
BASE_FOLDER = r"templates\player"

# Папка с шаблонами (разные варианты) крыс
RAT_TEMPLATES_FOLDER = r"templates\enemies\rat"

# Словарь поз игрока → (x_shift, y_shift) для определения «центра» персонажа
# POSE_OFFSETS = {
    # "RIGHT_UP":    (-1, 34),
    # "RIGHT":       (0, 18),
    # "RIGHT_DOWN":  (5, 36),
    # "LEFT_DOWN":   (5, 37),
    # "LEFT":        (0, 38),
    # "LEFT_UP":     (0, 36)
# }
POSE_OFFSETS = {
    "RIGHT_UP":    (0, 36),
    "RIGHT":       (0, 36),
    "RIGHT_DOWN":  (0, 36),
    "LEFT_DOWN":   (0, 36),
    "LEFT":        (0, 36),
    "LEFT_UP":     (0, 36)
}

# Пути к шаблонам иконок (выход из боя, в бою, ход противника, статистика боя)
EXIT_ICON =        r"templates\other\exit_fight_icon.bmp"
IN_BATTLE_ICON =   r"templates\other\in_battle_icon.bmp"
ENEMY_TURN_ICON =  r"templates\other\enemy_turn.bmp"
STATS_POPUP_ICON = r"templates\other\stats_popup_icon.bmp"

# Количество ударов по противнику
N_HITS = 2

# Клавиша мыши, которой проводится атака
# ATTACK_BTN = 'right'
ATTACK_BTN = 'left'

LAST_PLAYER_STEP = 32

MIN_ACTION_POINT = 5

# Глобальный флаг для управления основным циклом
RUNNING = True
STOP_AFTER_BATTLE = False

FIRST_TURN = True

###############################################################################
#                     ГЛОБАЛЬНЫЕ КОНСТАНТЫ РАСПОЗНАВАНИЯ ЦИФР                 #
###############################################################################

# Папка с шаблонами цифр, где лежат файлы "0.png", "1.png", ..., "9.png"
DIGITS_FOLDER = r"templates\digits"

# Глобальная константа (словарь). Заполняется один раз в init_digit_templates().
DIGIT_TEMPLATES = {}

# Путь к референсному изображению, по которому ищем "блок", рядом с которым есть число
REFERENCE_TEMPLATE_PATH = r"templates\other\ref_for_recognize.bmp"

# Смещение относительно верхнего левого угла найденного референса
SHIFT_X = 46
SHIFT_Y = 0

# Размер области, где нарисовано число (36×15)
NUM_WIDTH = 36
NUM_HEIGHT = 15

# Порог для поиска референса
REF_THRESHOLD = 0.85

# Порог для распознавания цифр
DIGIT_THRESHOLD = 0.8

###############################################################################
#                            ГЛОБАЛЬНЫЕ РЕСУРСЫ                               #
###############################################################################

# PLAYER_POSES_DICT: dict[str, list[tuple[str, np.ndarray]]]
#   { "pose_name": [(filename, gray_image), (filename2, gray_image2), ...], ... }
PLAYER_POSES_DICT = {}

# RAT_TEMPLATES: list[tuple[str, np.ndarray]]
#   [ (filename, gray_image), (filename2, gray_image2), ... ]
RAT_TEMPLATES = []

###############################################################################
#                      ЗАГРУЗКА РЕСУРСОВ В ПАМЯТЬ                             #
###############################################################################

def init_resources():
    """
    Однократная инициализация ресурсов:
      - Загружаем все шаблоны поз игрока (grayscale) в PLAYER_POSES_DICT.
      - Загружаем все шаблоны крыс (grayscale) в RAT_TEMPLATES.
    """
    global PLAYER_POSES_DICT, RAT_TEMPLATES

    # 1) Загружаем позы игрока
    PLAYER_POSES_DICT = load_player_poses_as_gray(BASE_FOLDER)
    logging.info(f"Загружено поз (папок): {list(PLAYER_POSES_DICT.keys())}")

    # 2) Загружаем крысиные шаблоны
    RAT_TEMPLATES = load_rat_templates_as_gray(RAT_TEMPLATES_FOLDER)
    logging.info(f"Загружено шаблонов крыс: {len(RAT_TEMPLATES)}")
    
    # Инициализация шаблонов для распознавания цифр
    init_digit_templates()

def load_player_poses_as_gray(base_folder):
    """
    Сканирует папку base_folder, где каждое имя подпапки считается названием позы.
    Внутри подпапки ищем файлы-изображения (bmp/png/jpg/jpeg).
    
    Для каждой позы создаётся список кортежей (filename, gray_image).
    Возвращает словарь:
      {
         pose_name1: [(fname1, img_gray1), (fname2, img_gray2), ...],
         pose_name2: [...]
      }
    """
    poses_dict = {}
    for folder_name in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(subfolder_path):
            continue

        pose_name = folder_name
        templates_list = []
        for file_name in os.listdir(subfolder_path):
            if file_name.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                full_path = os.path.join(subfolder_path, file_name)
                img_bgr = cv2.imread(full_path)
                if img_bgr is not None:
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    templates_list.append((file_name, img_gray))
                else:
                    logging.warning(f"Не удалось загрузить файл позы: {full_path}")

        if templates_list:
            poses_dict[pose_name] = templates_list

    return poses_dict


def load_rat_templates_as_gray(folder):
    """
    Сканирует заданную папку, ищет файлы-изображения (bmp/png/jpg/jpeg).
    Для каждого файла загружает в grayscale.
    
    Возвращает список кортежей (filename, gray_image).
    """
    templates = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            full_path = os.path.join(folder, fname)
            img_bgr = cv2.imread(full_path)
            if img_bgr is not None:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                templates.append((fname, img_gray))
            else:
                logging.warning(f"Не удалось загрузить файл: {full_path}")
    return templates


###############################################################################
#                          ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ШАБЛОНОВ
###############################################################################

def init_digit_templates():
    """
    Загружает шаблоны цифр (0..9) из DIGITS_FOLDER в глобальную переменную
    DIGIT_TEMPLATES, в градациях серого.
    """
    global DIGIT_TEMPLATES
    digit_templates = {}

    for d in range(10):
        fname = f"{d}.bmp"  # или .bmp/.jpg, если нужно
        full_path = os.path.join(DIGITS_FOLDER, fname)
        if not os.path.exists(full_path):
            logging.warning(f"Файл для цифры {d} не найден: {full_path}")
            continue

        img_bgr = cv2.imread(full_path)
        if img_bgr is None:
            logging.warning(f"Не удалось загрузить файл цифры: {full_path}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        digit_templates[d] = img_gray

    DIGIT_TEMPLATES = digit_templates
    logging.info(f"Загружено шаблонов цифр: {len(DIGIT_TEMPLATES)}")

###############################################################################
#                ФУНКЦИЯ ПОИСКА РЕФЕРЕНСА НА ЭКРАНЕ
###############################################################################

def find_reference_on_screen(ref_path, threshold=0.85):
    """
    Ищет заданный шаблон (ref_path) на всём экране, возвращает (x, y)
    при max_val >= threshold, иначе None.
    """
    screenshot = pyautogui.screenshot()
    screen_np = np.array(screenshot)
    screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    ref_bgr = cv2.imread(ref_path)
    if ref_bgr is None:
        logging.warning(f"Не удалось загрузить референс: {ref_path}")
        return None

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(screen_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    # logging.info(f"Поиск референса: max_val={max_val:.3f}, threshold={threshold}")

    if max_val >= threshold:
        # logging.info(f"Референс найден. Координаты: {max_loc}, val={max_val:.3f}")
        return max_loc
    else:
        logging.info("Референс не найден или max_val ниже порога.")
        return None

###############################################################################
#             РАСПОЗНАВАНИЕ 1..3 ЦИФР В ОБЛАСТИ 36×15 (GRAYSCALE)
###############################################################################

def recognize_digits_3chars(screen_gray_36x15, threshold=0.8):
    """
    Пытается распознать число (1..3 цифры) в фрагменте 36×15 (grayscale),
    используя глобальные DIGIT_TEMPLATES.
    Возвращает int или None.
    """
    global DIGIT_TEMPLATES

    h, w = screen_gray_36x15.shape
    if (w != 36 or h != 15):
        raise ValueError(f"Ожидалось изображение 36×15, а получено {w}×{h}")

    # 3 потенциальные позиции цифр
    digit_slices = [(0, 10), (13, 23), (26, 36)]
    recognized_digits = []

    for (sx, ex) in digit_slices:
        digit_sub = screen_gray_36x15[:, sx:ex]  # вырез 10×15
        best_val = 0
        best_digit = None

        for d, tpl_gray in DIGIT_TEMPLATES.items():
            th, tw = tpl_gray.shape
            if th != 15 or tw != 10:
                logging.warning(f"Шаблон цифры {d} имеет неожиданный размер: {tw}×{th}")
                continue

            result = cv2.matchTemplate(digit_sub, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_digit = d

        if best_val >= threshold and best_digit is not None:
            recognized_digits.append(best_digit)
        else:
            recognized_digits.append(None)

    # Удаляем ведущие None
    while recognized_digits and recognized_digits[0] is None:
        recognized_digits.pop(0)
    # Удаляем хвостовые None
    while recognized_digits and recognized_digits[-1] is None:
        recognized_digits.pop()

    digits_only = [d for d in recognized_digits if d is not None]
    if not digits_only:
        return None

    number_str = "".join(str(d) for d in digits_only)
    return int(number_str)

###############################################################################
#                           ПОЛУЧЕНИЕ ОЧКОВ ДЕЙСТВИЯ
###############################################################################

def get_action_points():
    """
    1) Ищет на экране референсное изображение (REFERENCE_TEMPLATE_PATH).
    2) Смещается на SHIFT_X, SHIFT_Y и снимает 36×15,
       где отображаются очки действия.
    3) Вызывает recognize_digits_3chars(...).
    4) Возвращает int (число очков действия) или None, если не найдено/не распознано.
    """
    ref_loc = find_reference_on_screen(REFERENCE_TEMPLATE_PATH, threshold=REF_THRESHOLD)
    if ref_loc is None:
        logging.warning("Не удалось найти референс. Очки действия не распознаны.")
        return None

    ref_x, ref_y = ref_loc
    num_x = ref_x + SHIFT_X
    num_y = ref_y + SHIFT_Y

    region = (num_x, num_y, NUM_WIDTH, NUM_HEIGHT)
    # logging.info(f"Снимаем скриншот области: {region}")

    screenshot = pyautogui.screenshot(region=region)
    screen_np = np.array(screenshot)
    screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    result = recognize_digits_3chars(screen_gray, threshold=DIGIT_THRESHOLD)
    if result is not None:
        logging.info(f"Распознано очков действия: {result}")
    else:
        logging.warning("Не удалось распознать число в области.")

    return result

###############################################################################
#                         ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                              #
###############################################################################

def _take_screenshot_region(region):
    """
    Делает скриншот указанной области экрана (region=(x, y, width, height))
    и возвращает BGR-изображение (numpy-массив).
    """
    screenshot = pyautogui.screenshot(region=region)
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    return screenshot_bgr


def take_screenshot_region(region, max_retries=3, delay=1):
    """
    Делает скриншот указанной области экрана (region=(x, y, width, height))
    и возвращает BGR-изображение (numpy-массив).
    
    Если возникает OSError (например, "screen grab failed"),
    повторно пытается сделать скриншот до max_retries раз с задержкой delay (секунд).
    Если все попытки неудачны, выбрасывает последнее исключение.
    
    :param region: (left, top, width, height)
    :param max_retries: сколько раз пытаться
    :param delay: пауза в секундах между попытками
    :return: numpy-массив (BGR) изображения
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            screenshot = pyautogui.screenshot(region=region)
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            return screenshot_bgr
        except OSError as e:
            # Ловим ошибку "screen grab failed" и пробуем снова
            logging.warning(f"Попытка {attempt} сделать скриншот не удалась: {e}")
            last_exception = e
            time.sleep(delay)
    # Если дошли сюда, значит все попытки исчерпаны
    logging.error(f"Не удалось сделать скриншот после {max_retries} попыток.")
    raise last_exception  # выбрасываем последнее исключение

def match_exists(screen_gray, template_gray, threshold=0.8):
    """
    Проверяет методом matchTemplate, есть ли совпадение template_gray в screen_gray
    с коэффициентом >= threshold.
    Возвращает True/False.
    
    :param screen_gray: np.ndarray, ч/б изображение (скриншот)
    :param template_gray: np.ndarray, ч/б шаблон
    """
    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return (max_val >= threshold)

###############################################################################
#              ПОИСК ПЕРСОНАЖА (ИГРОКА) ПО ЗАГРУЖЕННЫМ ШАБЛОНАМ               #
###############################################################################

def find_character_in_image(region, poses_dict, threshold=0.8):
    """
    Ищет персонажа по словарю поз (ключ=имя позы -> список (filename, gray_image)).
    Возвращает (best_loc, best_pose, best_val), где:
      - best_loc: (x, y) координаты ЛУЧШЕГО совпадения (в локальной системе region)
      - best_pose: название позы (или None)
      - best_val: float, максимально найденный match value
    """
    screen_bgr = take_screenshot_region(region)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    best_val = 0
    best_loc = None
    best_pose = None

    for pose_name, templates_list in poses_dict.items():
        for (fname, tpl_gray) in templates_list:
            # Сопоставляем
            result = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc_ = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc_
                best_pose = pose_name

    if best_val >= threshold and best_pose is not None:
        return best_loc, best_pose, best_val
    else:
        return None, None, 0

def get_player_coords(region):
    """
    Определяет координаты центра игрока:
      1) Ищет совпадение с лучшей позой (через PLAYER_POSES_DICT).
      2) Если найдено, добавляет соответствующий offset из POSE_OFFSETS,
         переводит локальную позицию в абсолютную и возвращает (x, y).
      3) Если не найдено, возвращает (None, None).
    """
    best_loc, best_pose, best_val = find_character_in_image(region, PLAYER_POSES_DICT, threshold=0.8)
    if best_loc is None:
        return None, None
    
    abs_x = region[0] + best_loc[0]
    abs_y = region[1] + best_loc[1]
    x_shift, y_shift = POSE_OFFSETS.get(best_pose, (0, 0))
    center_abs_x = abs_x + x_shift
    center_abs_y = abs_y + y_shift

    logging.info(f"Персонаж найден: поза={best_pose}, "
                 f"({center_abs_x},{center_abs_y}), val={best_val:.3f}")
    return center_abs_x, center_abs_y

###############################################################################
#                       ПОИСК КРЫС ПО ЗАГРУЖЕННЫМ ШАБЛОНАМ                     #
###############################################################################

def find_all_rat_enemies_in_region(region, rat_templates, threshold=0.8):
    """
    Ищет все совпадения крыс в заданной области (region).
      - rat_templates: список (filename, gray_image), загруженный заранее.
      - threshold: порог корреляции.

    Возвращает список кортежей (abs_x, abs_y, tpl_filename, match_val).
    """
    screen_bgr = take_screenshot_region(region)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    found_results = []
    for (fname, tpl_gray) in rat_templates:
        result = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for y, x in zip(loc[0], loc[1]):
            abs_x = region[0] + x
            abs_y = region[1] + y
            match_val = result[y, x]
            found_results.append((abs_x, abs_y, fname, match_val))

    return found_results

###############################################################################
#                   РАЗДЕЛЕНИЕ ПРОТИВНИКОВ НА "В ЗОНЕ" И "ВНЕ"                #
###############################################################################

def separate_enemies_by_rectangle(player_x, player_y, enemies_coords,
                                  attack_zone_width, attack_zone_height):
    """
    Разделяет список координат противников на 2 группы:
      - in_zone: те, что попадают в прямоугольник (attack_zone_width x attack_zone_height),
                 центрированный на (player_x, player_y).
      - out_zone: те, что не попадают.
    
    Также возвращает nearest_outside — ближайший из out_zone (или None, если его нет).
    
    :return: (in_zone, out_zone, nearest_outside)
    """
    in_zone = []
    out_zone = []

    left = player_x - attack_zone_width / 2
    right = player_x + attack_zone_width / 2
    top = player_y - attack_zone_height / 2
    bottom = player_y + attack_zone_height / 2

    for (ex, ey) in enemies_coords:
        ix = int(ex)
        iy = int(ey)
        if left <= ix <= right and top <= iy <= bottom:
            in_zone.append((ix, iy))
        else:
            out_zone.append((ix, iy))

    nearest_outside = None
    if out_zone:
        nearest_outside = min(
            out_zone,
            key=lambda e: (e[0] - player_x) ** 2 + (e[1] - player_y) ** 2
        )

    return in_zone, out_zone, nearest_outside


def is_point_in_polygon(px, py, polygon_points):
    """
    Проверка, лежит ли точка (px, py) внутри многоугольника (polygon_points)
    методом "ray casting".
    
    polygon_points: список вершин [(x0, y0), (x1, y1), ..., (xN, yN)]
    в порядке обхода (по часовой или против часовой стрелки).
    
    Возвращает True/False.
    """
    inside = False
    n = len(polygon_points)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        # Проверяем, пересекает ли луч "налево" полигон между yi и yj
        intersect = ((yi > py) != (yj > py)) and \
                    (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
    return inside

def separate_enemies_by_hex(player_x, 
                            player_y, 
                            enemies_coords,
                            zone_width, 
                            zone_height, 
                            extra_width):
    """
    Разделяет список координат противников на 2 группы (in_zone, out_zone),
    используя шестиугольную область вокруг (player_x, player_y).

    Параметры:
      - zone_width, zone_height: задают "центральный прямоугольник" (width × height),
        центрированный на (player_x, player_y).
      - extra_width: насколько шестиугольник «выпирает» треугольниками слева и справа.
        (Можно трактовать как "половину" ширины дополнительных треугольников.)

    Логика шестиугольника (6 вершин), расположенных примерно так:

         A ------ B
         |        |
       F |        | C
         |        |
         E ------ D

    где:
      A = (left,     top)
      B = (right,    top)
      C = (right + extra_width, center_y)
      D = (right,    bottom)
      E = (left,     bottom)
      F = (left - extra_width,  center_y)

    Возвращает (in_zone, out_zone, nearest_outside).
    """
    # Вычисляем вспомогательные координаты
    left   = player_x - zone_width / 2
    right  = player_x + zone_width / 2
    top    = player_y - zone_height / 2
    bottom = player_y + zone_height / 2
    # center_y = (top + bottom) / 2  # середина по вертикали
    center_y = player_y

    # Определяем 6 вершин шестиугольника в порядке обхода
    # (по часовой стрелке, начиная с верхней левой)
    A = (left,  top)
    B = (right, top)
    C = (right + extra_width, center_y)
    D = (right, bottom)
    E = (left,  bottom)
    F = (left - extra_width, center_y)

    hex_points = [A, B, C, D, E, F]  # 6 вершин

    in_zone = []
    out_zone = []

    # Проверяем каждую точку enemies_coords
    for (ex, ey) in enemies_coords:
        # Если лежит внутри шестиугольника - in_zone, иначе out_zone
        if is_point_in_polygon(ex, ey, hex_points):
            in_zone.append((ex, ey))
        else:
            out_zone.append((ex, ey))

    # Ищем ближайшего из out_zone к игроку
    nearest_outside = None
    if out_zone:
        nearest_outside = min(
            out_zone,
            key=lambda e: (e[0] - player_x) ** 2 + (e[1] - player_y) ** 2
        )

    return in_zone, out_zone, nearest_outside


###############################################################################
#                        УПРАВЛЕНИЕ ПЕРСОНАЖЕМ И КАМЕРОЙ                      #
###############################################################################

def shift_click(x, y, button='left'):
    """
    Зажимает SHIFT, кликает мышью по (x, y), затем отпускает SHIFT.
    """
    pydirectinput.keyDown(KEY_SHIFT)
    time.sleep(0.1)

    pydirectinput.moveTo(x, y)
    pydirectinput.click(button=button)

    pydirectinput.keyUp(KEY_SHIFT)
    time.sleep(0.1)


def cancel_popups():
    """
    Смахивает всплывающие окна (ящики ресурсов, выход из боя и т.п.)
    клавишами BACKSPACE и ESC.
    """
    pydirectinput.press(KEY_BACKSPACE)
    time.sleep(0.1)
    pydirectinput.press(KEY_ESC)
    time.sleep(0.1)


def focus_game_window():
    """
    Пример: клик в точку (70,70), чтобы вернуть фокус окну игры, 
    затем возвращаем курсор на старое место.
    """
    # x, y = pyautogui.position()
    pydirectinput.moveTo(80, 80)
    pydirectinput.click()
    # pydirectinput.moveTo(x, y)


def start_battle():
    """
    Имитируем нажатие Delete для начала боя (по вашему описанию).
    """
    focus_game_window()
    pydirectinput.keyDown(KEY_DELETE)
    time.sleep(0.1)
    pydirectinput.keyUp(KEY_DELETE)


def go_to_enemy_ext(player_x, player_y, enemy_x, enemy_y, num_clicks=5):
    """
    Пытается переместиться к противнику, делая несколько SHIFT-кликов вокруг (enemy_x, enemy_y).
    Проверяем после каждого клика, сместился ли игрок.
    """
    logging.info(f"Попытка переместиться к противнику: {enemy_x},{enemy_y}")
    # rnd_shuffle_x = [-36, 36, -34, 38, -36, 36, -34, 38]
    # rnd_shuffle_y = [-18, 18, -15, 20, -18, 18, -15, 20]
    rnd_shuffle_x = [-34, 34]
    rnd_shuffle_y = [-16, 16]
    
    # Если уже на месте
    if (player_x == enemy_x) and (player_y == enemy_y):
        pydirectinput.moveTo(player_x, player_y)
        pydirectinput.click()
        return

    for i in range(num_clicks):
        shift_click(
            enemy_x + random.choice(rnd_shuffle_x),
            enemy_y + random.choice(rnd_shuffle_y)
        )
        cancel_popups()

        px_new, py_new = get_player_coords(REGION)
        # if px_new is not None and py_new is not None:
            # dist_sqr = (px_new - player_x)**2 + (py_new - player_y)**2
            # if dist_sqr > 10:
                # logging.info("Персонаж переместился, успех!")
                # return

        # Если у игрока сменились координаты, значит задача выполнена
        if (px_new != player_x) or (py_new != player_y):
            logging.info("Персонаж переместился, это успех!")
            return

    logging.warning("Не удалось переместиться к противнику. Сдвиг не зафиксирован.")

###############################################################################
#                  ФУНКЦИИ ПРОВЕРКИ СТАТУСА БОЯ / ВЫХОДА ИЗ БОЯ               #
###############################################################################

def is_exit_fight(region=(1440, 340, 200, 100), exit_fight_path=EXIT_ICON, threshold=0.8):
    """
    Проверяет, есть ли на экране иконка "выход из боя".
    Возвращает True/False.
    """
    screen_bgr = take_screenshot_region(region)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    tpl_bgr = cv2.imread(exit_fight_path)
    if tpl_bgr is None:
        logging.warning(f"Не удалось загрузить шаблон 'выход из боя': {exit_fight_path}")
        return False
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

    return match_exists(screen_gray, tpl_gray, threshold=threshold)


def is_stats_popup(region=(500, 300, 600, 300), stats_popup_path=STATS_POPUP_ICON, threshold=0.8):
    """
    Проверяет, есть ли на экране иконка "итоги боя" (статистика).
    Возвращает True/False.
    """
    screen_bgr = take_screenshot_region(region)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    tpl_bgr = cv2.imread(stats_popup_path)
    if tpl_bgr is None:
        logging.warning(f"Не удалось загрузить шаблон 'итоги боя': {stats_popup_path}")
        return False
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

    return match_exists(screen_gray, tpl_gray, threshold=threshold)


def check_battle_status(region=(1650, 200, 210, 300), in_battle_path=IN_BATTLE_ICON, enemy_turn_path=ENEMY_TURN_ICON, threshold=0.8):
    """
    Проверяет статус боя:
      - None, если найдена иконка "ход противника";
      - True, если найдена иконка "в бою";
      - False, если ни одна из двух иконок не найдена (персонаж не в бою).
    """
    screen_bgr = take_screenshot_region(region)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    tpl_in_battle_bgr = cv2.imread(in_battle_path)
    tpl_enemy_turn_bgr = cv2.imread(enemy_turn_path)
    if tpl_in_battle_bgr is None or tpl_enemy_turn_bgr is None:
        logging.warning("Проблема с загрузкой иконок 'в бою' или 'ход противника'.")
        return False

    tpl_in_battle_gray = cv2.cvtColor(tpl_in_battle_bgr, cv2.COLOR_BGR2GRAY)
    tpl_enemy_turn_gray = cv2.cvtColor(tpl_enemy_turn_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Ход противника?
    if match_exists(screen_gray, tpl_enemy_turn_gray, threshold=threshold):
        return None

    # 2) В бою?
    if match_exists(screen_gray, tpl_in_battle_gray, threshold=threshold):
        return True

    # 3) Не в бою
    return False


def exit_battle():
    """
    Пытается несколько раз выйти из боя (клавишей '0') и закрыть окно статистики.
    """
    for i in range(5):
        # 1650, 200 - > 1860, 500 : (1650, 200, 210, 300) потому что (x, y, width, height)
        # status = check_battle_status(REGION)
        # status = check_battle_status((1650, 200, 210, 300)) 
        # status = check_battle_status()
        if check_battle_status() is not False:
            focus_game_window()
            
            pydirectinput.press(KEY_UNDO)
            time.sleep(0.1)
            
            pydirectinput.keyDown(KEY_EXIT_BAT)
            time.sleep(0.1)
            pydirectinput.keyUp(KEY_EXIT_BAT)

            pydirectinput.keyDown(KEY_EXIT_BAT)
            time.sleep(0.1)
            pydirectinput.keyUp(KEY_EXIT_BAT)
            
            time.sleep(2)

            # Несколько попыток найти окно статистики
            for j in range(5):
                # pydirectinput.keyDown(KEY_ENTER)
                # time.sleep(0.1)
                # pydirectinput.keyUp(KEY_ENTER)
                logging.info(f"Проверяем окно окончания боя, попытка {j}")
                if is_stats_popup():
                    logging.info("Закрываем окно статистики (Enter).")
                    pydirectinput.keyDown(KEY_ENTER)
                    time.sleep(0.1)
                    pydirectinput.keyUp(KEY_ENTER)
                    time.sleep(0.5)
                    break
                time.sleep(1)
        else:
            logging.info("Персонаж не в бою (exit_battle).")
            break
        time.sleep(0.5)

###############################################################################
#                            ЛОГИКА ХОДОВ / БОЯ                                #
###############################################################################

def attack_enemies(enemies_list):
    """
    Атакует каждого противника из списка enemies_list путём нескольких ударов (N_HITS).
    После каждого удара пытается отменить всплывающие окна (cancel_popups).
    """
    for (x, y) in enemies_list:
        for _ in range(N_HITS):
            pydirectinput.moveTo(x + 5, y + 5)
            pydirectinput.click(button=ATTACK_BTN)
        cancel_popups()

def get_all_enemies_around_player(player_x, player_y,
                                    attack_zone_width, attack_zone_height):
    # Ищем крыс
    enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.93)
    if not enemies:
        logging.info("Крыс не обнаружено в зоне!")
        return None, None, None
    else:
        logging.info(f"Найдено {len(enemies)} совпадений (крыс).")
        enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

        in_zone, out_zone, nearest_outside = separate_enemies_by_rectangle(
            player_x, player_y,
            enemy_coords,
            attack_zone_width,
            attack_zone_height
        )

        # logging.info(f"Противники в зоне: {in_zone}")
        # logging.info(f"Противники вне зоны: {out_zone}")
        # logging.info(f"Ближайший (вне зоны): {nearest_outside}")
        
        logging.info(f"Противников в зоне: {len(in_zone)}, вне зоны: {len(out_zone)}, ближайший (вне зоны): {nearest_outside if nearest_outside is not None else 'не найден!'}")
        # logging.info(f"Противников вне зоны: {len(out_zone)}")
        # logging.info(f"Ближайший (вне зоны): {nearest_outside}")
        return in_zone, out_zone, nearest_outside

def perform_battle_turn_old(attack_zone_width=330, attack_zone_height=320):
    """
    Совершает один 'ход' в бою:
      1) Фокус на окне, жмём Enter (закрыть окошко, если есть).
      2) Центрируем камеру (пробел).
      3) Находим игрока, ищем крыс, делим на ближних/дальних.
      4) Атакуем ближних, двигаемся к ближайшему дальнему (если есть).
      5) Нажимаем D, завершаем ход.
    """
    focus_game_window()
    pydirectinput.keyDown(KEY_ENTER)
    time.sleep(0.1)
    pydirectinput.keyUp(KEY_ENTER)

    time.sleep(0.2)

    # pydirectinput.press(KEY_SPACE)  # центрируем камеру
    # time.sleep(1)

    # Отменяем старый маршрут
    pydirectinput.press(KEY_UNDO)
    time.sleep(0.1)

    # pydirectinput.moveTo(0, 0)

    # # Перезарядка оружия
    # pydirectinput.press(KEY_RELOAD_W)
    # time.sleep(0.1)

    """Базовый метод"""
    # player_x, player_y = get_player_coords(REGION)
    # if player_x is None:
        # logging.warning("Персонаж не обнаружен на поле боя!")
        # focus_game_window()
        # time.sleep(0.1)
        # logging.info("Завершаем ход (D).")
        # pydirectinput.press(KEY_D)
        # return

    # in_zone, out_zone, nearest_outside = get_all_enemies_around_player(
        # player_x, player_y,
        # attack_zone_width,
        # attack_zone_height
    # )

    # if in_zone is not None:
        # # Атакуем ближних
        # attack_enemies(in_zone)

    # # Идём к самому близкому дальнему
    # if nearest_outside is not None:
        # go_to_enemy_ext(player_x, player_y,
                        # nearest_outside[0] + 5, nearest_outside[1] + 5)

        # # Пробуем получить очки действия
        # ap = get_action_points()
        # if (ap is not None) and (ap > 2):
            # logging.info("Пробуем снова найти игрока и противников вокруг!")

            # player_x, player_y = get_player_coords(REGION)
            # if player_x is not None:
                # in_zone_new, _, _ = get_all_enemies_around_player(
                    # player_x, player_y,
                    # attack_zone_width,
                    # attack_zone_height
                # )

                # if in_zone_new is not None:
                    # in_zone_new = list(set(in_zone_new) - set(in_zone)) 
                    # logging.info("Снова атакуем ближних!")
                    # # Атакуем ближних
                    # attack_enemies(in_zone_new)
        # else:
            # logging.info("Слишком мало очков действия для дополнительной атаки!")
    # else:
        # logging.info("Вне зоны атаки противников нет.")

    """Новый метод с циклом"""
    
    # 1. Ищем всех врагов до цикла
    
    # # Ищем крыс
    enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.93)
        
    if enemies:
        logging.info(f"Найдено {len(enemies)} совпадений (противников).")
        enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

        in_zone_old = []
        # Цикл совершения действий за один ход
        for i in range (5):
            logging.info(f"Связка атака+подшаг № {i+1}!")

            player_x, player_y = get_player_coords(REGION)
            if player_x is None:
                logging.warning("Персонаж не обнаружен на поле боя!")
                focus_game_window()
                time.sleep(0.1)
                logging.info("Завершаем ход (D).")
                pydirectinput.press(KEY_D)
                return

            # in_zone, out_zone, nearest_outside = get_all_enemies_around_player(
                # player_x, player_y,
                # attack_zone_width,
                # attack_zone_height
            # )
            # Определяем монстров около игрока 
            in_zone, out_zone, nearest_outside = separate_enemies_by_rectangle(
                player_x, player_y,
                enemy_coords,
                attack_zone_width,
                attack_zone_height
            )
            # Проверяем количество ОД для проведения атаки
            ap = get_action_points()
            if (ap is not None) and ap < MIN_ACTION_POINT:
                logging.warning("Недостаточно ОД для проведения атаки!")
                break
            else:
                logging.info("Можно атаковать!")

            if in_zone is not None:
                # in_zone = list(set(in_zone) - set(in_zone_old))

                # Атакуем ближних
                attack_enemies(in_zone)
                
                #### Тут нужно удалять из списка тех, кто был сейчас атакован
                enemy_coords = list(set(enemy_coords) - set(in_zone))
                
                # in_zone_old = in_zone
                # del in_zone

            # Проверяем количество ОД для прохода к противнику
            ap = get_action_points()
            if (ap is not None) and ap < LAST_PLAYER_STEP:
                logging.warning("Недостаточно ОД для прохода к противнику!")
                # центрируем камеру
                pydirectinput.press(KEY_SPACE)

                # # Перезарядка оружия
                # pydirectinput.press(KEY_RELOAD_W)
                # time.sleep(0.1)
                break
            else:
                logging.info("Можно идти к противнику!")

            # Идём к самому близкому дальнему
            if nearest_outside is not None:
                go_to_enemy_ext(player_x, player_y,
                                nearest_outside[0] + 5, nearest_outside[1] + 5)
            else:
                logging.warning("За зоной атаки противников не найдено, идти некуда!")
                break

            # Проверяем количество ОД для проведения атаки
            ap = get_action_points()
            if (ap is not None) and ap < MIN_ACTION_POINT:
                logging.warning("Недостаточно ОД для проведения атаки!")
                break
            else:
                logging.info("Пробуем очередную связку атака+подшаг!")
    else:
        logging.info("Крыс не обнаружено в зоне!")    

    #####################################   
    # # Ищем крыс
    # enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.95)
    # if not enemies:
        # logging.info("Крыс не обнаружено в зоне!")
    # else:
        # logging.info(f"Найдено {len(enemies)} совпадений (крыс).")
        # enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

        # in_zone, out_zone, nearest_outside = separate_enemies_by_rectangle(
            # player_x, player_y,
            # enemy_coords,
            # attack_zone_width,
            # attack_zone_height
        # )

        # logging.info(f"Противники в зоне: {in_zone}")
        # logging.info(f"Противники вне зоны: {out_zone}")
        # logging.info(f"Ближайший (вне зоны): {nearest_outside}")

        # # Атакуем ближних
        # attack_enemies(in_zone)

        # # Идём к самому близкому дальнему
        # if nearest_outside is not None:
            # go_to_enemy_ext(player_x, player_y,
                            # nearest_outside[0] + 5, nearest_outside[1] + 5)
        # else:
            # logging.info("Вне зоны атаки противников нет.")
        
        # # 2) Пробуем получить очки действия
        # ap = get_action_points()
        # if ap > 3:
            # player_x, player_y = get_player_coords(REGION)

            # if player_x is not None:
                # # Ищем крыс
                # enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.95)
                # if not enemies:
                    # logging.info("Крыс не обнаружено в зоне!")
                # else:
                    # logging.info(f"Найдено {len(enemies)} совпадений (крыс).")
                    # enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

                    # in_zone, out_zone, nearest_outside = separate_enemies_by_rectangle(
                        # player_x, player_y,
                        # enemy_coords,
                        # attack_zone_width,
                        # attack_zone_height
                    # )

                    # logging.info(f"Противники в зоне: {in_zone}")
                    # logging.info(f"Противники вне зоны: {out_zone}")
                    # logging.info(f"Ближайший (вне зоны): {nearest_outside}")

                    # # Атакуем ближних
                    # attack_enemies(in_zone)          
    
    focus_game_window()
    time.sleep(0.1)

    # Перезарядка оружия
    pydirectinput.press(KEY_RELOAD_W)
    time.sleep(0.1)

    logging.info("Завершаем ход в штатном режиме (D).")
    pydirectinput.press(KEY_D)

def perform_battle_turn(attack_zone_width=330, attack_zone_height=320):
    """
    Совершает один 'ход' в бою:
      1) Фокус на окне, жмём Enter (закрыть окошко, если есть).
      2) Центрируем камеру (пробел).
      3) Находим игрока, ищем крыс, делим на ближних/дальних.
      4) Атакуем ближних, двигаемся к ближайшему дальнему (если есть).
      5) Нажимаем D, завершаем ход.
    """
    global FIRST_TURN

    focus_game_window()
    pydirectinput.keyDown(KEY_ENTER)
    time.sleep(0.1)
    pydirectinput.keyUp(KEY_ENTER)

    time.sleep(0.2)

    # центрируем камеру
    # pydirectinput.press(KEY_SPACE)
    # time.sleep(1)

    # Отменяем старый маршрут
    pydirectinput.press(KEY_UNDO)
    time.sleep(0.1)

    # # Перезарядка оружия
    # pydirectinput.press(KEY_RELOAD_W)
    # time.sleep(0.1)

    if FIRST_TURN:
        FIRST_TURN = False
        # Выкинуть 4 секцию рюкзака
        pydirectinput.press(KEY_DROP_ITEM)
        time.sleep(0.1)
        pydirectinput.press(KEY_ENTER)

    """Новый метод с циклом"""
    
    # 1. Ищем всех врагов до цикла
    
    enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.93)
        
    if enemies:
        logging.info(f"Всего обнаружено {len(enemies)} противников.")

        # Пересобираем массив, оставляя только координаты (x, y)
        enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

        # Цикл совершения действий (шагов) за один ход
        for i in range (5):
            if not RUNNING:
                return

            logging.info(f"Связка атаки и прохода к противнику № {i+1}!")

            # Проверяем количество ОД для проведения атаки
            ap = get_action_points()
            if (ap is not None) and ap < MIN_ACTION_POINT:
                logging.warning(f"Недостаточное количество ОД для дальнейших действий!")
                break
            else:
                logging.info(f"ОД хватает ({ap}) для дальнейших действий.")

            # Определяем координаты игрока
            player_x, player_y = get_player_coords(REGION)
            if player_x is None:
                logging.warning("Персонаж не обнаружен на поле боя!")
                focus_game_window()
                time.sleep(0.1)
                logging.info("Завершаем ход (D).")
                pydirectinput.press(KEY_D)
                return

            # Определяем монстров в заданной зоне около игрока 
            # in_zone, out_zone, nearest_outside = separate_enemies_by_rectangle(
                # player_x, player_y,
                # enemy_coords,
                # attack_zone_width,
                # attack_zone_height
            # )
            in_zone, out_zone, nearest_outside = separate_enemies_by_hex(
                player_x, player_y,
                enemy_coords,
                attack_zone_width,
                attack_zone_height,
                extra_width=150
            )
            if not RUNNING:
                return

            # Атакуем близлежащих противников (если они найдены)
            if in_zone is not None:
                attack_enemies(in_zone)
                
                # Удаляем из массива тех противников, кто был атакован
                enemy_coords = list(set(enemy_coords) - set(in_zone))

            # Проверяем количество ОД для прохода к противнику
            ap = get_action_points()
            if (ap is not None) and ap < LAST_PLAYER_STEP:
                logging.warning("Недостаточно ОД для прохода к противнику!")
                # центрируем камеру
                pydirectinput.press(KEY_SPACE)

                # # Перезарядка оружия
                # pydirectinput.press(KEY_RELOAD_W)
                # time.sleep(0.1)
                break
            else:
                logging.info(f"ОД хватает ({ap}), чтобы идти к противнику!")

            # Идём к самому близкому за зоной атаки
            if not RUNNING:
                return
    
            # Добавочная логика:
            # Пробуем сделать шаг,
                # если изменилась координата, то ОК,
                # если координата не изменилась, но ушли ОД, значит отменить шаг
            if nearest_outside is not None:
                go_to_enemy_ext(player_x, player_y,
                                nearest_outside[0] + 5, nearest_outside[1] + 5)
            else:
                logging.warning("За зоной атаки противников не найдено, идти некуда!")
                break
    else:
        logging.info("Крыс не обнаружено в зоне!")    

    focus_game_window()
    time.sleep(0.1)

    # Перезарядка оружия
    pydirectinput.press(KEY_RELOAD_W)
    time.sleep(0.1)

    logging.info("Завершаем ход в штатном режиме (D).")
    pydirectinput.press(KEY_D)

def test_attack(attack_zone_width=330, attack_zone_height=320, attack_zone_extra_width=150):
    # 1. Ищем всех врагов
    enemies = find_all_rat_enemies_in_region(REGION, RAT_TEMPLATES, threshold=0.93)

    if enemies:
        logging.info(f"Всего обнаружено {len(enemies)} противников.")

        # Пересобираем массив, оставляя только координаты (x, y)
        enemy_coords = [(ex, ey) for (ex, ey, _, _) in enemies]

        # Определяем координаты игрока
        player_x, player_y = get_player_coords(REGION)
        if player_x is None:
            logging.warning("Персонаж не обнаружен на поле боя!")
            focus_game_window()
            time.sleep(0.1)
            logging.info("Завершаем ход (D).")
            pydirectinput.press(KEY_D)
            return

        in_zone, out_zone, nearest_outside = separate_enemies_by_hex(
            player_x, player_y,
            enemy_coords,
            attack_zone_width,
            attack_zone_height,
            attack_zone_extra_width
        )
        # Атакуем близлежащих противников (если они найдены)
        if in_zone is not None:
            attack_enemies(in_zone)
    else:
        logging.info("Крыс не обнаружено в зоне!") 


def process_battle():
    """
    Цикл, пока персонаж в бою:
      - Если найдена иконка выхода из боя, пытаемся выйти (exit_battle).
      - Иначе делаем ход (perform_battle_turn).
      - Если check_battle_status -> False, значит персонаж вышел из боя.
    """
    global FIRST_TURN
    FIRST_TURN = True

    while True:
        # status = check_battle_status()
        if check_battle_status() is False:
            # Персонаж уже не в бою
            break
        else:
            if is_exit_fight():
                exit_battle()
                break
            else:
                if not RUNNING:
                    return
                perform_battle_turn()
        time.sleep(0.5)

###############################################################################
#                           ОСНОВНОЙ ЦИКЛ РАБОТЫ                              #
###############################################################################

def stop_script():
    """
    Немедленная остановка скрипта: устанавливает RUNNING в False.
    """
    global RUNNING
    logging.warning("Получен сигнал немедленной остановки скрипта!")
    RUNNING = False

def stop_after_battle():
    """
    Устанавливает флаг остановки после окончания текущего боя.
    Скрипт завершится сразу после завершения текущей итерации боя.
    """
    global STOP_AFTER_BATTLE
    logging.warning("Получен сигнал остановки скрипта после боя!")
    STOP_AFTER_BATTLE = True

def _main_loop():
    """
    Основной бесконечный цикл:
      1) Проверяем, не в бою ли персонаж:
         - Если нет, начинаем бой (start_battle).
         - Если да, обрабатываем бой (process_battle).
      2) Повторяем, пока RUNNING=True.
    """
    global RUNNING
    logging.info("Запуск основного цикла... (нажмите Ctrl+Shift+Z для остановки)")

    battle_attempt = 1
    total_battles = 0

    while RUNNING:
        logging.info(f"Завершено боёв: {total_battles}")

        # На всякий случай жмём ENTER — закрыть окно окончания боя, если висит
        pydirectinput.press(KEY_ENTER)

        logging.info(f"Попытка начать бой № {battle_attempt}")

        # Проверяем статус боя
        if check_battle_status() == False:
            # Не в бою — жмём Delete для начала
            start_battle()
            time.sleep(3)
        else:
            # Уже в бою — обрабатываем его
            process_battle()
            total_battles += 1

        battle_attempt += 1

    logging.warning("Цикл остановлен, завершаем скрипт.")


def __main_loop(battles_limit=0):
    """
    Основной цикл:
      1) Если персонаж не в бою – начинаем бой (start_battle).
      2) Если в бою – process_battle().
      3) Ограничение по числу боёв (battles_limit), если > 0.
    """
    global RUNNING
    logging.info("Запуск основного цикла... (нажмите Ctrl+Shift+Z для остановки)")

    battle_attempt = 1
    total_battles = 0

    while RUNNING:
        # Проверяем, не достигнут ли лимит
        if battles_limit > 0 and total_battles >= battles_limit:
            logging.info(f"Достигнут лимит боёв: {battles_limit}. Останавливаемся.")
            break

        logging.info(f"Завершено боёв: {total_battles}")

        # На всякий случай жмём ENTER — закрыть окно окончания боя, если висит
        pydirectinput.press(KEY_ENTER)

        logging.info(f"Попытка начать бой № {battle_attempt}")

        # Проверяем статус боя
        if check_battle_status() == False:
            # Не в бою — жмём Delete для начала
            start_battle()
            time.sleep(3)
        else:
            # Уже в бою — обрабатываем его
            process_battle()
            total_battles += 1

        battle_attempt += 1

    logging.warning("Цикл остановлен, завершаем скрипт.")

###############################################################################
#                            ОСНОВНОЙ ЦИКЛ РАБОТЫ
###############################################################################

def main_loop(battles_limit=0):
    """
    Основной цикл работы скрипта:
      - Если персонаж не в бою, начинает бой (start_battle).
      - Если персонаж в бою, обрабатывает бой (process_battle).
      - Если установлен флаг STOP_AFTER_BATTLE, завершает цикл после текущего боя.
      - Если задан лимит боёв (> 0) и их достигнуто, цикл завершает работу.
      
    :param battles_limit: int, количество боёв до остановки (0 – безлимит)
    """
    global RUNNING, STOP_AFTER_BATTLE

    logging.info(f"Запуск основного цикла... ({KEY_STOP_SCRIPT} – немедленная остановка, {KEY_STOP_SCRIPT_AFTER_BATTLE} – остановка после боя)")
    battle_attempt = 1
    total_battles = 0

    while RUNNING:
        logging.info(f"Завершено боёв: {total_battles}")
        # Если задан лимит боёв и он достигнут, завершаем цикл
        if battles_limit > 0 and total_battles >= battles_limit:
            logging.info(f"Достигнут лимит боёв: {battles_limit}. Останавливаем скрипт.")
            break
        
        # На всякий случай жмём ENTER, чтобы закрыть возможное всплывающее окно
        pydirectinput.press(KEY_ENTER)

        logging.info(f"Попытка начать бой № {battle_attempt}")

        # Проверяем, в бою ли персонаж
        if not check_battle_status():
            start_battle()
            time.sleep(3)
        else:
            process_battle()
            total_battles += 1
            # Если флаг остановки после боя установлен, выходим
            if STOP_AFTER_BATTLE:
                logging.warning("Остановка после боя по запросу. Завершаем цикл.")
                RUNNING = False
                break

        battle_attempt += 1

    logging.warning("Основной цикл завершён. Скрипт остановлен.")


def _main():
    """
    Точка входа в программу:
      1) Инициализируем ресурсы (загружаем шаблоны игрока и крыс в память).
      2) Регистрируем горячую клавишу (Ctrl+Shift+Z) для прерывания работы.
      3) Запускаем основной цикл.
    """
    init_resources()  # Однократная загрузка иконок / поз / крыс в память

    # Регистрируем хоткей остановки
    # keyboard.add_hotkey('ctrl+shift+z', stop_script)
    keyboard.add_hotkey(KEY_STOP_SCRIPT, stop_script)

    # Запускаем основной цикл
    main_loop()


def __main():
    """
    Точка входа. Спрашиваем у пользователя, сколько боёв нужно провести (0 – безлимитно).
    Затем запускаем основной цикл.
    """
    init_resources()  # например, загрузка шаблонов и т.п.
    # keyboard.add_hotkey(os.environ.get("KEY_STOP_SCRIPT", "ctrl+shift+z"), stop_script)
    keyboard.add_hotkey(KEY_STOP_SCRIPT, stop_script)
    
    # Спрашиваем у пользователя лимит боёв
    battles_limit_str = input("Введите число боёв, по достижении которого скрипт должен остановиться (0 – безлимит): ")
    try:
        battles_limit = int(battles_limit_str)
    except ValueError:
        logging.warning("Некорректный ввод. Будет использовано значение 0 (безлимит).")
        battles_limit = 0

    main_loop(battles_limit)

###############################################################################
#                                MAIN
###############################################################################

def main():
    """
    Точка входа:
      0. Инициализируем ресурсы (загружаем шаблоны игрока и крыс в память).
      1. Регистрируем горячие клавиши для остановки.
      2. Запрашиваем у пользователя лимит боёв.
      3. Запускаем основной цикл.
    """

    init_resources()  # Однократная загрузка иконок / поз / крыс в память

    # Регистрируем глобальные хоткеи
    keyboard.add_hotkey(KEY_STOP_SCRIPT, stop_script)
    keyboard.add_hotkey(KEY_STOP_SCRIPT_AFTER_BATTLE, stop_after_battle)

    # Запрашиваем лимит боёв у пользователя
    battles_limit_str = input("Введите число боёв (0 для безлимитного режима): ")
    try:
        battles_limit = int(battles_limit_str)
    except ValueError:
        logging.warning("Некорректный ввод. Будет использовано 0 (безлимит).")
        battles_limit = 0

    main_loop(battles_limit)

if __name__ == "__main__":
    main()
    
    # init_resources()
    # test_attack()
    # get_player_coords(REGION)
