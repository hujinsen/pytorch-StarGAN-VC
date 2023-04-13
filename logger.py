import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.create_file_writer(log_dir)
        # changed FileWriter to create_file_writer as Tensorlow v1 changed to v2

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        # changed the add_summary with reagrd to tensorlow v2. working perfectly
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)