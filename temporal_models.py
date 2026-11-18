"""Temporal order-preserving architectures; inputs always stop at decision time."""
def build_temporal_model(cfg,architecture):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(cfg.seed)
    tf.config.experimental.enable_op_determinism()
    L=tf.keras.layers
    inputs=L.Input((cfg.history_rows,3))
    if architecture=='tcn':
        x=L.AveragePooling1D(pool_size=4,strides=4)(inputs)
        x=L.Conv1D(24,1)(x)
        for dilation in [1,2,4,8,16]:
            residual=x
            z=L.Conv1D(24,3,dilation_rate=dilation,padding='causal',activation='relu')(x)
            z=L.LayerNormalization(axis=-1)(z)
            z=L.SpatialDropout1D(.15)(z)
            z=L.Conv1D(24,3,dilation_rate=dilation,padding='causal')(z)
            x=L.Activation('relu')(L.Add()([residual,z]))
        # Select final time step without a Lambda, for safe standard serialization.
        x=L.Cropping1D((int(x.shape[1])-1,0))(x)
        x=L.Flatten()(x)
    elif architecture=='gru':
        x=L.Conv1D(24,7,strides=4,padding='causal',activation='relu')(inputs)
        x=L.GRU(32)(x)
    else:raise ValueError('Unknown temporal architecture')
    x=L.Dropout(.25)(x)
    outputs=L.Dense(1,activation='sigmoid')(x)
    model=tf.keras.Model(inputs,outputs,name='ankle_'+architecture+'_strict')
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate,clipnorm=1.),
                  loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
    return model
