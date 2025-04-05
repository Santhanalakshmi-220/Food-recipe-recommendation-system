import streamlit as st

import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer

from PIL import (
    ImageFont,
)

import os
import re
import random
import textwrap
from examples import EXAMPLES
import dummy
import meta
from utils import ext
from utils.api import generate_cook_image
from utils.draw import generate_food_with_logo_image, generate_recipe_image
from utils.st import (
    remote_css,
    local_css,

)
from utils.utils import (
    load_image_from_url,
    load_image_from_local,
    image_to_base64,
    pure_comma_separation
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TextGeneration:
    def __init__(self):
        self.debug = False
        self.dummy_outputs = dummy.recipes
        self.tokenizer = None
        self.generator = None
        self.api_ids = []
        self.api_keys = []
        self.api_test = 2
        self.task = "text2text-generation"
        self.model_name_or_path = "flax-community/t5-recipe-generation"
        self.color_frame = "#ffffff"
        self.main_frame = os.path.join(BASE_DIR, "asset", "frame", "recipe-bg.png")
        self.no_food = os.path.join(BASE_DIR, "asset", "frame", "no_food.png")
        self.logo_frame = os.path.join(BASE_DIR, "asset", "frame", "logo1.png")
        self.chef_frames = {
            "scheherazade": os.path.join(BASE_DIR, "asset", "frame", "food-image-logo-bg-s.png"),
            "giovanni": os.path.join(BASE_DIR, "asset", "frame", "food-image-logo-bg-g.png"),
        }
        self.fonts = {
             "title": ImageFont.truetype(os.path.join(BASE_DIR, "asset", "fonts", "Poppins-Bold.ttf"), 70),
             "sub_title": ImageFont.truetype(os.path.join(BASE_DIR, "asset", "fonts", "Poppins-Medium.ttf"), 30),
             "body_bold": ImageFont.truetype(os.path.join(BASE_DIR, "asset", "fonts", "Montserrat-Bold.ttf"), 22),
             "body": ImageFont.truetype(os.path.join(BASE_DIR, "asset", "fonts", "Montserrat-Regular.ttf"), 18),
         }
        set_seed(42)

    def _skip_special_tokens_and_prettify(self, text):
        recipe_maps = {"<sep>": "--", "<section>": "\n"}
        recipe_map_pattern = "|".join(map(re.escape, recipe_maps.keys()))

        text = re.sub(
            recipe_map_pattern,
            lambda m: recipe_maps[m.group()],
            re.sub("|".join(self.tokenizer.all_special_tokens), "", text)
        )

        data = {"title": "", "ingredients": [], "directions": []}
        for section in text.split("\n"):
            section = section.strip()
            if section.startswith("title:"):
                data["title"] = " ".join(
                    [w.strip().capitalize() for w in section.replace("title:", "").strip().split() if w.strip()]
                )
            elif section.startswith("ingredients:"):
                data["ingredients"] = [s.strip() for s in section.replace("ingredients:", "").split('--')]
            elif section.startswith("directions:"):
                data["directions"] = [s.strip() for s in section.replace("directions:", "").split('--')]
            else:
                pass

        return data

    def load_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.generator = pipeline(self.task, model=self.model_name_or_path, tokenizer=self.model_name_or_path)

    def load_api(self):
        app_ids = os.getenv("EDAMAM_APP_ID")
        app_ids = app_ids.split(",") if app_ids else []
        app_keys = os.getenv("EDAMAM_APP_KEY")
        app_keys = app_keys.split(",") if app_keys else []

        if len(app_ids) != len(app_keys):
            self.api_ids = []
            self.api_keys = []

        self.api_ids = app_ids
        self.api_keys = app_keys

    def load(self):
        self.load_api()
        if not self.debug:
            self.load_pipeline()

    def prepare_frame(self, recipe, chef_name):
        frame_path = self.chef_frames[chef_name.lower()]
        food_logo = generate_food_with_logo_image(frame_path, self.logo_frame, recipe["image"])
        frame = generate_recipe_image(
            recipe,
            self.main_frame,
            food_logo,
            self.fonts,
            bg_color="#ffffff"
        )
        return frame

    def generate(self, items, generation_kwargs):
        recipe = self.dummy_outputs[0]

        if not self.debug:
            generation_kwargs["num_return_sequences"] = 1
            generation_kwargs["return_tensors"] = True
            generation_kwargs["return_text"] = False

            generated_ids = self.generator(
                items,
                **generation_kwargs,
            )[0]["generated_token_ids"]
            recipe = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            recipe = self._skip_special_tokens_and_prettify(recipe)

        if self.api_ids and self.api_keys and len(self.api_ids) == len(self.api_keys):
            test = 0
            for i in range(len(self.api_keys)):
                if test > self.api_test:
                    recipe["image"] = None
                    break
                image = generate_cook_image(recipe["title"].lower(), self.api_ids[i], self.api_keys[i])
                test += 1
                if image:
                    recipe["image"] = image
                    break
        else:
            recipe["image"] = None

        return recipe

    def generate_frame(self, recipe, chef_name):
        return self.prepare_frame(recipe, chef_name)


@st.cache_resource()
def load_text_generator():
    generator = TextGeneration()
    generator.load()
    return generator
