import json

# -----------------------------
# Paths
# -----------------------------
TRAIN_JSON = "data/annotations/instances_train2017.json"
VAL_JSON = "data/annotations/instances_val2017.json"

TRAIN_FILES = "train_files.txt"
VAL_FILES = "val_files.txt"

# COCO category ID -> Our label
LABEL_MAP = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    6: 3,   # bus
    18: 4   # dog
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_file_list(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f)


train_data = load_json(TRAIN_JSON)
val_data = load_json(VAL_JSON)

train_files = load_file_list(TRAIN_FILES)
val_files = load_file_list(VAL_FILES)


def create_labels(dataset, selected_files):

    # image_id -> filename
    image_map = {}

    for image in dataset["images"]:
        if image["file_name"] in selected_files:
            image_map[image["id"]] = image["file_name"]

    labels = {}

    for ann in dataset["annotations"]:

        if ann["image_id"] in image_map:

            category = ann["category_id"]

            if category in LABEL_MAP:

                filename = image_map[ann["image_id"]]

                # Save only the first valid label
                if filename not in labels:
                    labels[filename] = LABEL_MAP[category]

    return labels


train_labels = create_labels(train_data, train_files)
val_labels = create_labels(val_data, val_files)

with open("train_labels.json", "w") as f:
    json.dump(train_labels, f, indent=4)

with open("val_labels.json", "w") as f:
    json.dump(val_labels, f, indent=4)

print("Training labels:", len(train_labels))
print("Validation labels:", len(val_labels))
print("Label files created successfully!")