The author said that before using his program, users have to modify four files contained in this folder. Since keras updates constantly, replacing these files would cause problems. Hence, I list detailed modifications here for others to modify corresponding files directly.

### < keras DIR >\layers\convolutional.py

Add the following new definition:

```
class MyUpSampling2D(Layer):

    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), num_pixels = (0, 0), data_format=None, **kwargs):
        super(MyUpSampling2D, self).__init__(**kwargs)
        """
        Note:
        	For python 3, this could better be 
        	super().__init__(**kwargs)
        	num_pixels = (0, 0) is not exist in the original version
        """
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)
        self.num_pixels = num_pixels
        """
        Note:
        	self.size indicate the upsampling rate.
        """
    """
    Note:
    	self.num_pixels = num_pixels is not in the original version
    """

    def compute_output_shape(self, input_shape):
        """
        Note:
        	+ self.num_pixels[0/1] does not exist in the original version
        	It seems that the author add a symetrical boarder to the output tensor.
        """
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] + self.num_pixels[0] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] + self.num_pixels[1] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] + self.num_pixels[0] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] + self.num_pixels[1] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        return K.resize_images(inputs, self.size[0], self.size[1],
                               self.data_format, self.num_pixels)
    """
    Note:
    	The last parameter self.num_pixels does not exist in the original version.
    """

    def get_config(self):
        """
        Note:
          The last item in config does not exist in the original version.
        """
        config = {'size': self.size,
                  'data_format': self.data_format,
                  'num_pixels': self.num_pixels}
        base_config = super(MyUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

### < keras DIR >\backend\tensorflow_backend.py

Modify the following new definition:

```
def resize_images(x, height_factor, width_factor, data_format, num_pixels=None):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    """
    Note:
    	Although the last parameter num_pixels does not exist in the original version, the new version is compatible with the original one.
    """
    if data_format == 'channels_first':
        original_shape = int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = int_shape(x) # (None, 67, 90, 512)
        new_shape = tf.shape(x)[1:3] # (67, 90, 512)
        #print(new_shape.get_shape().as_list())
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        if(num_pixels is not None):
           new_shape += tf.constant(np.array([num_pixels[0], num_pixels[1]]).astype('int32'))
        
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        if(num_pixels is not None):
            x.set_shape((None, original_shape[1] * height_factor + num_pixels[0] if original_shape[1] is not None else None,
                     original_shape[2] * width_factor + num_pixels[1] if original_shape[2] is not None else None, None))
        else:
            x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)
```

### < keras DIR >\losses.py

Modify the following definition:

```
import tensorflow as tf # this is newly imported

def binary_crossentropy(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1]) # pass [-1] to flatten tensor
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    """
    Note:
    	This definition is not the same as the original one. But this definition is competible with the original version.
    """
```

### < keras DIR >\metrics.py

Modify the following definition:

```
import tensorflow as tf

def binary_accuracy(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    """
    Note:
    	This definition is different from, but competible with the original one.
    """
```

### <PYTHON 3.6 DIR>\site-packages\skimage\transform\pyramids.py

replace

```
out_rows = math.ceil(rows / float(downscale))
out_cols = math.ceil(cols / float(downscale))
```

with

```
out_rows = math.floor(rows / float(downscale))
out_cols = math.floor(cols / float(downscale))
```