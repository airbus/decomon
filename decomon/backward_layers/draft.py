def call_previous(self, inputs):
    x = inputs[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

    weight, bias = self.layer.get_backward_weights(x)

    if self.activation_name != "linear":
        x_output = self.layer.call_linear(x)
        # here update x
        if self.finetune:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
                finetune=self.alpha_b_l,
            )
        else:

            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
            )

    weights = K.expand_dims(K.expand_dims(weight, -1), 0)  # (1, n_in, n_out, 1)
    bias = K.expand_dims(K.expand_dims(bias, -1), 0)  # (1, n_out, 1)

    b_out_u_ = K.sum(w_out_u * bias, 1) + b_out_u
    b_out_l_ = K.sum(w_out_l * bias, 1) + b_out_l

    w_out_u_ = K.sum(K.expand_dims(w_out_u, 1) * weights, 2)
    w_out_l_ = K.sum(K.expand_dims(w_out_l, 1) * weights, 2)

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_


def call_no_previous(self, inputs):
    x = inputs
    weight, bias = self.layer.get_backward_weights(x)

    if self.activation_name != "linear":
        # here update x
        x_output = self.layer.call_linear(x)
        y_ = x_output[-1]
        shape = np.prod(y_.shape[1:])

        if self.finetune:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output,
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
                finetune=self.alpha_b_l,
                previous=self.previous,
            )

        else:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output, convex_domain=self.convex_domain, slope=self.slope, mode=self.mode, previous=self.previous
            )
            w_out_u = K.reshape(w_out_u, (-1, shape))
            b_out_u = K.reshape(b_out_u, (-1, shape))
            w_out_l = K.reshape(w_out_l, (-1, shape))
            b_out_l = K.reshape(b_out_l, (-1, shape))

        weights = K.expand_dims(K.expand_dims(weight, -1), 0)  # (1, n_in, n_out, 1)
        bias = K.expand_dims(K.expand_dims(bias, -1), 0)  # (1, n_out, 1)

        b_out_u_ = K.sum(w_out_u * bias, 1) + b_out_u
        b_out_l_ = K.sum(w_out_l * bias, 1) + b_out_l

        w_out_u_ = K.sum(K.expand_dims(w_out_u, 1) * weights, 2)
        w_out_l_ = K.sum(K.expand_dims(w_out_l, 1) * weights, 2)


    else:
        z_value = K.cast(0.0, K.floatx())
        y_ = x[-1]
        shape = np.prod(y_.shape[1:])
        y_flatten = K.reshape(z_value * y_, (-1, np.prod(shape), 1))  # (None, n_in, 1)
        w_out_u_ = y_flatten + K.expand_dims(weight, 0)
        w_out_l_ = w_out_u_
        b_out_u_ = K.sum(y_flatten, 1) + K.expand_dims(bias, 0)
        b_out_l_ = b_out_u_

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_


def call_previous_old(self, inputs):
    x = inputs[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

    # infer the output dimension
    x_output = self.layer.call_linear(x)

    if self.activation_name != "linear":
        # here update x

        if self.finetune:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
                finetune=self.alpha_b_l,
            )

        else:

            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
            )

    output_shape_tensor = x_output[-1].shape
    n_out = w_out_u.shape[-1]
    # first permute dimensions
    # w_out_u = array_ops.transpose(w_out_u, perm=(0, 1, 3, 2))
    # w_out_l = array_ops.transpose(w_out_l, perm=(0, 1, 3, 2))

    w_out_u = array_ops.transpose(w_out_u, perm=(0, 2, 1))
    w_out_l = array_ops.transpose(w_out_l, perm=(0, 2, 1))

    if self.layer.data_format == "channels_last":
        height, width, channel = [e for e in output_shape_tensor[1:]]
        w_out_u = K.reshape(w_out_u, [-1, n_out, height, width, channel])
        w_out_l = K.reshape(w_out_l, [-1, n_out, height, width, channel])

        # start with bias
        if self.layer.use_bias:
            bias = self.layer.bias[None, None, None, None]  # (1, 1, 1, 1, channel)
            # w_out_u * bias : (None, n_out, height, width, channel)
            b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
            b_out_u_ += b_out_u
            b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
            b_out_l_ += b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l

    else:
        channel, height, width = [e for e in output_shape_tensor[1:]]
        w_out_u = K.reshape(w_out_u, [-1, n_out, channel, height, width])
        w_out_l = K.reshape(w_out_l, [-1, n_out, channel, height, width])

        # start with bias
        if self.layer.use_bias:
            bias = self.layer.bias[None, None, :, None, None]  # (1, 1, channel, 1, 1)
            # w_out_u * bias : (None, n_out, height, width, channel)
            b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
            b_out_u_ += b_out_u
            b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
            b_out_l_ += b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l

    """
    else:
        b_out_u_ = K.expand_dims(K.sum(w_out_u * bias, (2, 3, 4)), 1)
        b_out_l_ = K.expand_dims(K.sum(w_out_l * bias, (2, 3, 4)), 1)
    """

    kernel_ = array_ops.transpose(self.layer.kernel, (0, 1, 3, 2))

    w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0 = self.call_no_previous(x)

    # import pdb; pdb.set_trace()

    def step_func(z, _):
        return (
            conv2d_transpose(
                z,
                self.layer.kernel,
                x[-1].shape,
                strides=self.layer.strides,
                padding=self.layer.padding,
                data_format=self.layer.data_format,
                dilation_rate=self.layer.dilation_rate,
            ),
            [],
        )

    step_func(w_out_u[:, 0], 0)  # init
    w_out_u_ = K.rnn(step_function=step_func, inputs=w_out_u, initial_states=[], unroll=False)[1]
    w_out_l_ = K.rnn(step_function=step_func, inputs=w_out_l, initial_states=[], unroll=False)[1]

    n_in = np.prod(w_out_u_.shape[2:])
    w_out_u_ = array_ops.transpose(K.reshape(w_out_u_, [-1, n_out, n_in]), perm=(0, 2, 1))
    w_out_l_ = array_ops.transpose(K.reshape(w_out_l_, [-1, n_out, n_in]), perm=(0, 2, 1))

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_


def call_no_previous_old(self, inputs):
    if len(inputs):
        x = inputs
        # infer the output dimension
        x_output = self.layer.call_linear(x)
    else:
        x = self.layer.outputs
        x_output = self.layer.outputs
    y_ = x_output[-1]
    shape = np.prod(y_.shape[1:])

    if self.activation_name != "linear":
        # here update x

        if self.finetune:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output,
                convex_domain=self.convex_domain,
                slope=self.slope,
                mode=self.mode,
                finetune=self.alpha_b_l,
                previous=self.previous,
            )

        else:
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output, convex_domain=self.convex_domain, slope=self.slope, mode=self.mode, previous=self.previous
            )
            w_out_u = K.reshape(w_out_u, (-1, shape))
            b_out_u = K.reshape(b_out_u, (-1, shape))
            w_out_l = K.reshape(w_out_l, (-1, shape))
            b_out_l = K.reshape(b_out_l, (-1, shape))
    else:

        z_value = K.cast(0.0, K.floatx())
        o_value = K.cast(1.0, K.floatx())
        y_flat = K.reshape(y_, [-1, shape])

        w_out_u, w_out_l, b_out_u, b_out_l = [o_value + z_value * y_flat] * 4

    w_out_u = tf.linalg.diag(w_out_u)
    w_out_l = tf.linalg.diag(w_out_l)

    output_shape_tensor = x_output[-1].shape
    n_out = w_out_u.shape[-1]
    # first permute dimensions
    # w_out_u = array_ops.transpose(w_out_u, perm=(0, 1, 3, 2))
    # w_out_l = array_ops.transpose(w_out_l, perm=(0, 1, 3, 2))

    w_out_u = array_ops.transpose(w_out_u, perm=(0, 2, 1))
    w_out_l = array_ops.transpose(w_out_l, perm=(0, 2, 1))

    if self.layer.data_format == "channels_last":
        height, width, channel = [e for e in output_shape_tensor[1:]]
        w_out_u = K.reshape(w_out_u, [-1, n_out, height, width, channel])
        w_out_l = K.reshape(w_out_l, [-1, n_out, height, width, channel])

        # start with bias
        if self.layer.use_bias:
            bias = self.layer.bias[None, None, None, None]  # (1, 1, 1, 1, channel)
            # w_out_u * bias : (None, n_out, height, width, channel)
            b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
            b_out_u_ += b_out_u
            b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
            b_out_l_ += b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l

    else:
        channel, height, width = [e for e in output_shape_tensor[1:]]
        w_out_u = K.reshape(w_out_u, [-1, n_out, channel, height, width])
        w_out_l = K.reshape(w_out_l, [-1, n_out, channel, height, width])

        # start with bias
        if self.layer.use_bias:
            bias = self.layer.bias[None, None, :, None, None]  # (1, 1, channel, 1, 1)
            # w_out_u * bias : (None, n_out, height, width, channel)
            b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
            b_out_u_ += b_out_u
            b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
            b_out_l_ += b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l

    """
    else:
        b_out_u_ = K.expand_dims(K.sum(w_out_u * bias, (2, 3, 4)), 1)
        b_out_l_ = K.expand_dims(K.sum(w_out_l * bias, (2, 3, 4)), 1)
    """

    kernel_ = array_ops.transpose(self.layer.kernel, (0, 1, 3, 2))

    def step_func(z, _):
        return (
            conv2d_transpose(
                z,
                kernel_,
                x[-1].shape,
                strides=self.layer.strides,
                padding=self.layer.padding,
                data_format=self.layer.data_format,
                dilation_rate=self.layer.dilation_rate,
            ),
            [],
        )

    step_func(w_out_u[:, 0], 0)  # init
    w_out_u_ = K.rnn(step_function=step_func, inputs=w_out_u, initial_states=[], unroll=False)[1]
    w_out_l_ = K.rnn(step_function=step_func, inputs=w_out_l, initial_states=[], unroll=False)[1]

    n_in = np.prod(w_out_u_.shape[2:])
    w_out_u_ = array_ops.transpose(K.reshape(w_out_u_, [-1, n_out, n_in]), perm=(0, 2, 1))
    w_out_l_ = array_ops.transpose(K.reshape(w_out_l_, [-1, n_out, n_in]), perm=(0, 2, 1))

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_
