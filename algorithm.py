import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import BayesianOptimization
import json

############################################################
# CONFIGURAÇÕES INICIAIS
############################################################

DATASET_DIR = "dataset"  # diretório raiz do dataset
TRAIN_DIR = os.path.join(DATASET_DIR, "C:/Users/Daniel/Downloads/Gabriel/dataset/train")
VAL_DIR = os.path.join(DATASET_DIR, "C:/Users/Daniel/Downloads/Gabriel/dataset/validation")

# Critérios de iteração
INITIAL_TRIALS = 100
ADDITIONAL_TRIALS = 50
MAX_TRIALS_LIMIT = 500

IMPROVEMENT_THRESHOLD = 0.001  # 0.1% = 0.001 em termos relativos
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.10  # 10% = 0.10 melhoria significativa se chegar a 500 trials
STAGNATION_CHECK_TRIALS = 20
FINAL_THRESHOLD_TO_CONTINUE_AT_500 = SIGNIFICANT_IMPROVEMENT_THRESHOLD

# Critério de overfitting definido previamente
OVERFITTING_THRESHOLD = 0.10
tf.random.set_seed(42)

############################################################
# ETAPA 1: ANÁLISE E VERIFICAÇÃO DO DATASET
############################################################

train_classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
val_classes = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])

if train_classes != val_classes:
    raise ValueError("As classes em validation não correspondem às classes em train.")

num_classes = len(train_classes)
print("Número de classes detectadas:", num_classes)
print("Classes:", train_classes)

def count_images_in_dir(path):
    total = 0
    for c in os.listdir(path):
        c_path = os.path.join(path, c)
        if os.path.isdir(c_path):
            total += len([f for f in os.listdir(c_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    return total

print("Imagens em treinamento:", count_images_in_dir(TRAIN_DIR))
print("Imagens em validação:", count_images_in_dir(VAL_DIR))

############################################################
# FUNÇÃO DE CRIAÇÃO DO PIPELINE DE DADOS
############################################################

def get_datasets(img_size=(224,224), batch_size=32, color_mode='rgb',
                 rotation_factor=0.0, zoom_factor=0.0, horizontal_flip=False, vertical_flip=False, brightness_factor=0.0):
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=False
    )
    
    data_augmentation_layers = []
    if rotation_factor > 0:
        data_augmentation_layers.append(layers.RandomRotation(rotation_factor))
    if zoom_factor > 0:
        data_augmentation_layers.append(layers.RandomZoom(zoom_factor))
    if horizontal_flip:
        data_augmentation_layers.append(layers.RandomFlip("horizontal"))
    if vertical_flip:
        data_augmentation_layers.append(layers.RandomFlip("vertical"))
    if brightness_factor > 0:
        data_augmentation_layers.append(layers.RandomBrightness(brightness_factor))
    
    preprocessing = tf.keras.Sequential([
        layers.Rescaling(1./255)
    ])
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_augs = tf.keras.Sequential(data_augmentation_layers) if data_augmentation_layers else None
    
    def augment(x, y):
        if train_augs:
            x = train_augs(x, training=True)
        x = preprocessing(x)
        return x, y

    def preprocess_val(x, y):
        x = preprocessing(x)
        return x, y
    
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    
    return train_ds, val_ds

############################################################
# FUNÇÃO PARA CRIAR MODELO (CUSTOM OU TRANSFER)
############################################################

def build_model(hp):
    model_type = hp.Choice("model_type", ["custom", "ResNet50", "InceptionV3", "EfficientNetB0", "MobileNetV2", "DenseNet121", "VGG16"])
    input_shape = (hp.Choice("img_size", [128, 224, 256]),
                   hp.Choice("img_size", [128, 224, 256]),
                   3 if hp.Boolean("use_rgb") else 1)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    if model_type == "custom":
        conv_layers = hp.Int("conv_layers", 2, 5)
        filters = hp.Int("filters", 16, 64, step=16)
        activation = hp.Choice("activation", ["relu","elu","selu"])
        normalization = hp.Choice("normalization", ["batch","layer"])
        
        for i in range(conv_layers):
            x = layers.Conv2D(filters*(2**i), (3,3), padding='same')(x)
            if normalization == "batch":
                x = layers.BatchNormalization()(x)
            else:
                x = layers.LayerNormalization()(x)
            x = layers.Activation(activation)(x)
            if hp.Boolean("use_pooling"):
                x = layers.MaxPooling2D()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
    else:
        base_model = None
        if model_type == "ResNet50":
            base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == "InceptionV3":
            base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == "EfficientNetB0":
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == "MobileNetV2":
            base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == "DenseNet121":
            base_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        elif model_type == "VGG16":
            base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        total_layers = len(base_model.layers)
        trainable_layers = hp.Int("trainable_layers", 10, total_layers, step=10)
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)
    
    dense_layers = hp.Int("dense_layers", 0, 3)
    for _ in range(dense_layers):
        units = hp.Int("dense_units", 64, 512, step=64)
        x = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(hp.Float("l2", 1e-6, 1e-4, sampling="log")))(x)
        x = layers.Activation("relu")(x)
        if hp.Boolean("use_dropout"):
            x = layers.Dropout(hp.Float("dropout_rate", 0.0, 0.5, step=0.1))(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    
    optimizer_choice = hp.Choice("optimizer", ["adam","rmsprop","sgd"])
    lr = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")
    
    if optimizer_choice == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        momentum = hp.Float("momentum", 0.0, 0.9, step=0.1)
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

############################################################
# FUNÇÃO OBJETIVO PARA A OTIMIZAÇÃO
############################################################

def model_builder(hp):
    return build_model(hp)

def run_trial(hp):
    img_size = hp.get("img_size")
    use_rgb = hp.get("use_rgb")
    color_mode = "rgb" if use_rgb else "grayscale"
    batch_size = hp.Choice("batch_size", [8,16,32,64])
    rotation_factor = hp.Float("rotation", 0.0, 0.15, step=0.05)
    zoom_factor = hp.Float("zoom", 0.0, 0.2, step=0.05)
    horizontal_flip = hp.Boolean("hflip")
    vertical_flip = hp.Boolean("vflip")
    brightness_factor = hp.Float("brightness", 0.0, 0.1, step=0.05)
    
    train_ds, val_ds = get_datasets(
        img_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode=color_mode,
        rotation_factor=rotation_factor,
        zoom_factor=zoom_factor,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_factor=brightness_factor
    )
    
    model = build_model(hp)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    epochs = hp.Int("epochs", 10, 200, step=10)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )
    
    val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy'])
    train_acc = history.history['accuracy'][best_epoch]
    return val_acc, train_acc

############################################################
# IMPLEMENTAÇÃO DO CICLO ADAPTATIVO DE TRIALS
############################################################

def adaptive_search(tuner, initial_trials=100, add_trials=50, max_limit=500,
                    improvement_threshold=IMPROVEMENT_THRESHOLD,
                    stagnation_check=STAGNATION_CHECK_TRIALS,
                    significant_threshold=SIGNIFICANT_IMPROVEMENT_THRESHOLD):

    # Rodar os primeiros 100 trials
    current_max_trials = initial_trials
    tuner.search(max_trials=current_max_trials)
    
    best_scores = get_val_acc_history(tuner)
    # Função auxiliar para calcular se houve melhora
    # Retorna True se houve melhora relativa > improvement_threshold nos últimos 'check_len' trials
    # ou False se estagnou.
    def check_improvement(history, check_len=stagnation_check, threshold=improvement_threshold):
        if len(history) < check_len+1:
            return True  # Ainda não temos histórico suficiente, presumir que há possibilidade de melhora
        recent = history[-check_len:]
        prev = history[-check_len-1]
        # Verificar a porcentagem de melhora entre prev e o valor médio ou máximo dos recentes
        current_best = max(recent)
        relative_improvement = (current_best - prev) / (prev + 1e-8)
        return relative_improvement > threshold
    
    # Enquanto estiver melhorando, adiciona mais 50 trials
    while True:
        if check_improvement(best_scores, stagnation_check, improvement_threshold):
            # Está melhorando, rodar mais 50
            if current_max_trials >= max_limit:
                # Chegou no limite de 500 trials
                # Verificar se há melhora significativa nos últimos 50 trials
                if check_significant_improvement(best_scores, check_len=50, threshold=significant_threshold):
                    current_max_trials += add_trials
                    if current_max_trials > max_limit + add_trials:  
                        # Mesmo após a melhora significativa, não vamos continuar indefinidamente
                        break
                else:
                    # Não houve melhora significativa. Parar.
                    break
            else:
                current_max_trials += add_trials
        else:
            # Não está melhorando (estagnou por 20 trials)
            break
        
        # Rodar novamente o tuner com o novo limite
        tuner.search(max_trials=current_max_trials)
        best_scores = get_val_acc_history(tuner)
    
    # Ao final, temos o melhor conjunto de hiperparâmetros
    best_hp = get_best_hparams_no_overfit(tuner)
    if best_hp is None:
        # Se não encontrou um conjunto sem overfitting significativo
        best_hp = tuner.get_best_hyperparameters(1)[0]
    return best_hp


def get_val_acc_history(tuner):
    # Obter o histórico da melhor val_accuracy a cada trial concluído
    trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))
    val_accs = [t.metrics.get("val_accuracy", 0) for t in tuner.oracle.trials.values()]
    # Ordenar conforme a ordem em que foram rodados (trial_id)
    # trial_id em tuner.oracle.trials pode ser string, ordenar pelo tempo de criação
    sorted_trials = sorted(tuner.oracle.trials.values(), key=lambda x: x.start_time)
    return [tr.metrics.get("val_accuracy",0) for tr in sorted_trials if tr.status == "COMPLETED"]

def check_significant_improvement(history, check_len=50, threshold=0.10):
    # Verifica se houve aumento de pelo menos 10% (ou threshold) nos últimos 50 trials
    # em relação ao valor no início desses 50 trials
    if len(history) <= check_len:
        return True  # Se não há 50 trials, considerar que ainda há melhora
    recent = history[-check_len:]
    prev = history[-check_len-1]
    current_best = max(recent)
    relative_improvement = (current_best - prev) / (prev + 1e-8)
    return relative_improvement > threshold

def get_best_hparams_no_overfit(tuner):
    # Selecionar melhores hparams sem overfitting
    best_trials = sorted(tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)),
                         key=lambda t: t.metrics.get("val_accuracy", 0),
                         reverse=True)
    for t in best_trials:
        val_acc = t.metrics.get("val_accuracy")
        train_acc = t.metrics.get("train_accuracy")
        if train_acc is not None and val_acc is not None:
            if (train_acc - val_acc) < OVERFITTING_THRESHOLD:
                return t.hyperparameters
    return None

############################################################
# EXECUÇÃO DA BUSCA ADAPTATIVA
############################################################

class OverfittingCheckTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        val_acc, train_acc = run_trial(trial.hyperparameters)
        self.oracle.update_trial(trial.trial_id, {'val_accuracy': val_acc, 'train_accuracy': train_acc})
        self.oracle.save()

tuner = OverfittingCheckTuner(
    hypermodel=model_builder,
    objective="val_accuracy",
    directory="tuner_results",
    project_name="nobra_project",
    overwrite=True,
    max_trials=INITIAL_TRIALS
)

print("Iniciando a busca Bayesiana de hiperparâmetros com critério adaptativo...")
best_hparams = adaptive_search(tuner, initial_trials=INITIAL_TRIALS, 
                               add_trials=ADDITIONAL_TRIALS, 
                               max_limit=MAX_TRIALS_LIMIT,
                               improvement_threshold=IMPROVEMENT_THRESHOLD,
                               stagnation_check=STAGNATION_CHECK_TRIALS,
                               significant_threshold=SIGNIFICANT_IMPROVEMENT_THRESHOLD)

print("Melhores hiperparâmetros (sem overfitting) ou caso não encontrado, melhor global:", best_hparams.values)

############################################################
# TREINO FINAL DO MODELO COM HIPERPARÂMETROS SELECIONADOS
############################################################

final_hp = best_hparams
final_img_size = final_hp.get("img_size")
final_color_mode = "rgb" if final_hp.get("use_rgb") else "grayscale"
final_batch_size = final_hp.get("batch_size")
train_ds, val_ds = get_datasets(
    img_size=(final_img_size, final_img_size),
    batch_size=final_batch_size,
    color_mode=final_color_mode,
    rotation_factor=final_hp.get("rotation"),
    zoom_factor=final_hp.get("zoom"),
    horizontal_flip=final_hp.get("hflip"),
    vertical_flip=final_hp.get("vflip"),
    brightness_factor=final_hp.get("brightness")
)

final_model = build_model(final_hp)

final_epochs = final_hp.get("epochs")
early_stop_final = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Treinando modelo final...")
final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=final_epochs,
    callbacks=[early_stop_final],
    verbose=1
)

# Salvar modelo final
final_model.save("best_model.keras")
print("Modelo final salvo em best_model.keras")

# Salvar hiperparâmetros escolhidos
with open("best_hparams.json", "w") as f:
    json.dump(final_hp.values, f)
print("Hiperparâmetros salvos em best_hparams.json")
