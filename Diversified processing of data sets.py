import os
import time
from PIL import Image, ImageEnhance, ImageChops
from concurrent.futures import ThreadPoolExecutor


def extract_base_name_and_suffix(filename):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return base_name, ""


def image_reversal(img, savefilepath, base_name, suffix):
    lr = img.transpose(Image.FLIP_LEFT_RIGHT)
    ud = img.transpose(Image.FLIP_TOP_BOTTOM)
    lr.save(os.path.join(savefilepath, f"{base_name}_lr_{suffix}.png"))
    ud.save(os.path.join(savefilepath, f"{base_name}_ud_{suffix}.png"))


def image_rotation(img, savefilepath, base_name, suffix):
    out1 = img.rotate(40, expand=True)
    out2 = img.rotate(30, expand=True)
    out1.save(os.path.join(savefilepath, f"{base_name}_rot40_{suffix}.png"))
    out2.save(os.path.join(savefilepath, f"{base_name}_rot30_{suffix}.png"))


def image_translation(img, savefilepath, base_name, suffix):
    out_x = ImageChops.offset(img, 20, 0)
    out_y = ImageChops.offset(img, 0, 20)
    out_x.save(os.path.join(savefilepath, f"{base_name}_transx_{suffix}.png"))
    out_y.save(os.path.join(savefilepath, f"{base_name}_transy_{suffix}.png"))


def image_brightness(img, savefilepath, base_name, suffix):
    bri = ImageEnhance.Brightness(img)
    bri.enhance(0.8).save(os.path.join(savefilepath, f"{base_name}_bri08_{suffix}.png"))
    bri.enhance(1.2).save(os.path.join(savefilepath, f"{base_name}_bri12_{suffix}.png"))


def image_chroma(img, savefilepath, base_name, suffix):
    col = ImageEnhance.Color(img)
    col.enhance(0.7).save(os.path.join(savefilepath, f"{base_name}_col07_{suffix}.png"))
    col.enhance(1.3).save(os.path.join(savefilepath, f"{base_name}_col13_{suffix}.png"))


def image_contrast(img, savefilepath, base_name, suffix):
    con = ImageEnhance.Contrast(img)
    con.enhance(0.7).save(os.path.join(savefilepath, f"{base_name}_con07_{suffix}.png"))
    con.enhance(1.3).save(os.path.join(savefilepath, f"{base_name}_con13_{suffix}.png"))


def image_sharpness(img, savefilepath, base_name, suffix):
    sha = ImageEnhance.Sharpness(img)
    sha.enhance(0.5).save(os.path.join(savefilepath, f"{base_name}_sha05_{suffix}.png"))
    sha.enhance(1.5).save(os.path.join(savefilepath, f"{base_name}_sha15_{suffix}.png"))


def process_image(filepath, savefilepath, filename):
    image_path = os.path.join(filepath, filename)
    try:
        with Image.open(image_path) as img:
            if img.mode == "P":
                img = img.convert('RGB')
            base_name, suffix = extract_base_name_and_suffix(filename)
            image_reversal(img, savefilepath, base_name, suffix)
            image_rotation(img, savefilepath, base_name, suffix)
            image_translation(img, savefilepath, base_name, suffix)
            image_brightness(img, savefilepath, base_name, suffix)
            image_chroma(img, savefilepath, base_name, suffix)
            image_contrast(img, savefilepath, base_name, suffix)
            image_sharpness(img, savefilepath, base_name, suffix)

    except Exception as e:
        print(f": {image_path}, : {e}")


def image_expansion(filepath, savefilepath):
    """
    :param filepath:
    :param savefilepath:
    """
    if not os.path.exists(savefilepath):
        os.makedirs(savefilepath)

    filenames = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for filename in filenames:
            executor.submit(process_image, filepath, savefilepath, filename)


if __name__ == '__main__':
    filepath = 'D:/mypycharm/pythonProject5/log8/'
    savefilepath = 'D:/mypycharm/pythonProject5/log08/'

    start_time = time.time()
    image_expansion(filepath, savefilepath)
    elapsed_time = time.time() - start_time
    print(f'{elapsed_time:.2f}')
