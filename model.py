import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gradients_impl import gradients
from losses import dice_loss, rbox_loss

RESIZE_FACTOR = 2

def resize_bilinear(x):
  return tf.image.resize(x, size=[tf.shape(x)[1] * RESIZE_FACTOR, tf.shape(x)[2] * RESIZE_FACTOR])

def resize_output_shape(input_shape):
  shape = list(input_shape)
  assert len(shape) == 4
  shape[1] *= RESIZE_FACTOR
  shape[2] *= RESIZE_FACTOR
  return tuple(shape)

class EAST_model(tf.keras.Model):
  def __init__(self, input_size=512):
    super(EAST_model, self).__init__()

    self.score_map_loss_weight = tf.Variable(0.01, name='score_map_loss_weight')
    self.small_text_weight = tf.Variable(0., name='small_text_weight')

    self.compiled_loss = [dice_loss, rbox_loss]

    input_image = tf.keras.layers.Input(shape=(None, None, 3), name='input_image')
    resnet = tf.keras.applications.ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)
    x = resnet.get_layer('conv5_block3_out').output

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_1')(x)
    x = tf.keras.layers.concatenate([x, resnet.get_layer('conv4_block6_out').output], axis=3)
    x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_2')(x)
    x = tf.keras.layers.concatenate([x, resnet.get_layer('conv3_block4_out').output], axis=3)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_3')(x)
    x = tf.keras.layers.concatenate([x, resnet.get_layer('conv2_block3_out').output], axis=3)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    pred_score_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
    rbox_geo_map = tf.keras.layers.Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
    rbox_geo_map = tf.keras.layers.Lambda(lambda x: x * input_size)(rbox_geo_map)
    angle_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
    angle_map = tf.keras.layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
    pred_geo_map = tf.keras.layers.concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

    model = tf.keras.models.Model(inputs=[input_image], outputs=[pred_score_map, pred_geo_map])

    self.model = model
    self.input_image = input_image
    self.pred_score_map = pred_score_map
    self.pred_geo_map = pred_geo_map

  def train_step(self, data):
    [input_images, overly_small_text_region_training_masks, text_region_boundary_training_masks, _], \
      [target_score_maps, target_geo_maps] = data

    with tf.GradientTape() as tape:
      score_y_pred, geo_y_pred = self(input_images)
      _dice_loss = dice_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks, self.score_map_loss_weight,
                        self.small_text_weight, target_score_maps, score_y_pred)
      _rbox_loss = rbox_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks,
                        self.small_text_weight, target_score_maps, target_geo_maps, geo_y_pred)
      loss = _dice_loss + _rbox_loss
              
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return {'dice_loss':_dice_loss, 'rbox_loss': _rbox_loss, 'loss': loss}
  
  def test_step(self, data):
    [input_images, overly_small_text_region_training_masks, text_region_boundary_training_masks, _], \
      [target_score_maps, target_geo_maps] = data
    
    score_y_pred, geo_y_pred = self(input_images, training=False)
    _dice_loss = dice_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks, self.score_map_loss_weight,
                      self.small_text_weight, target_score_maps, score_y_pred)
    _rbox_loss = rbox_loss(overly_small_text_region_training_masks, text_region_boundary_training_masks,
                      self.small_text_weight, target_score_maps, target_geo_maps, geo_y_pred)
    loss = _dice_loss + _rbox_loss

    return {'dice_loss':_dice_loss, 'rbox_loss': _rbox_loss, 'loss': loss}

  def call(self, x):
    if isinstance(x, tuple):
      return self.model(x[0])
    else:
      return self.model(x)
