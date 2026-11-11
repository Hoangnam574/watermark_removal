import os
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


def main():
    # Load config
    FLAGS = ng.Config('inpaint.yml')

    # GPU config
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # Dataset (đọc theo cấu trúc neuralgym)
    dataset = ng.data.DataFromFNames(
        FLAGS.data_flist,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        repeat=True)

    batch_data = dataset.next()

    # Build model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        FLAGS,
        batch_data,
        training=True,
        summary=True,   
        reuse=False
    )

    # Optimizers
    global_step = tf.Variable(0, trainable=False, name='global_step')

    g_opt = tf.train.AdamOptimizer(
        learning_rate=FLAGS.lr,
        beta1=FLAGS.beta1,
        beta2=FLAGS.beta2
    )

    d_opt = tf.train.AdamOptimizer(
        learning_rate=FLAGS.lr,
        beta1=FLAGS.beta1,
        beta2=FLAGS.beta2
    )

    g_train = g_opt.minimize(
        losses['g_loss'],
        var_list=g_vars,
        global_step=global_step
    )

    d_train = d_opt.minimize(
        losses['d_loss'],
        var_list=d_vars
    )

    # Saver
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        # Resume training if checkpoint exists
        if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            print('Checkpoint restored.')

        for step in range(FLAGS.max_iters):
            # Train discriminator
            sess.run(d_train)

            # Train generator
            _, g_loss, ae_loss = sess.run(
                [g_train, losses['g_loss'], losses['ae_loss']]
            )

            if step % FLAGS.log_interval == 0:
                print(
                    f"Step {step} | "
                    f"G_loss: {g_loss:.4f} | "
                    f"AE_loss: {ae_loss:.4f}"
                )

            if step % FLAGS.save_interval == 0 and step > 0:
                saver.save(
                    sess,
                    os.path.join(FLAGS.checkpoint_dir, 'model'),
                    global_step=step
                )
                print('Checkpoint saved.')


if __name__ == "__main__":
    main()
