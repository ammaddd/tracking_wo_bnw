import logging
import functools
try:
    import comet_ml
    comet_installed = True
except:
    comet_installed = False


class CometLogger:
    def __init__(self, comet=False, **kwargs):
        global comet_installed
        self._logging = None
        self._comet_args = kwargs
        if comet == False:
            self._logging = False
        elif comet == True and comet_installed == False:
            raise Exception("Comet not installed. Run 'pip install comet-ml'")
    
    def _requiresComet(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self = args[0]
            global comet_installed
            if self._logging is None and comet_installed:
                self._logging = False
                try:
                    comet_ml.init()
                    if comet_ml.get_global_experiment() is not None:
                        logging.warning("You have already created a comet \
                                        experiment manually, which might \
                                        cause clashes")
                    self._experiment = comet_ml.Experiment(**self._comet_args)
                    self._logging = True
                except Exception as e:
                    logging.warning(e)

            if self._logging == True:
                return method(*args, **kwargs)
        return wrapper
    
    @_requiresComet
    def end(self):
        """Ends an experiment."""
        self._experiment.end()
        comet_ml.config.experiment = None

    @_requiresComet
    def log_metric(self, name, value, step=None, epoch=None,
                   include_context=True):
        """Logs a general metric (i.e accuracy, f1)..

        Args:
            name: String - Name of your metric
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
            include_context: Optional. If set to True (the default),
                the current context will be logged along the metric.
        """
        self._experiment.log_metric(name, value, step, epoch,
                                    include_context)

    @_requiresComet
    def log_model(self, name, file_or_folder, file_name=None):
        """Logs the model data under the name. Data can be a file path,
            a folder path or a file-like object.

        Args:         
            name: string (required), the name of the model
            file_or_folder: the model data (required); can be a file path,
                a folder path or a file-like object.
            file_name: (optional) the name of the model data. Used with.
        """
        self._experiment.log_model(name, file_or_folder, file_name)

    @_requiresComet
    def log_image(self, image_data, name=None, overwrite=False,
                  image_format="png", image_scale=1.0, image_shape=None,
                  image_colormap=None, image_minmax=None, image_channels="last",
                  copy_to_tmp=True, step=None):
        """Logs the image. Images are displayed on the Graphics tab on Comet.ml.

        Args:     
            image_data: Required. image_data is one of the following:
                a path (string) to an image
                a file-like object containing an image
                a numpy matrix
                a TensorFlow tensor
                a PyTorch tensor
                a list or tuple of values
                a PIL Image
            name: String - Optional. A custom name to be displayed on
                the dashboard. If not provided the filename from the
                image_data argument will be used if it is a path.
            overwrite: Optional. Boolean - If another image with the
                same name exists, it will be overwritten if overwrite
                is set to True.
            image_format: Optional. String. Default: 'png'. If the 
                image_data is actually something that can be turned
                into an image, this is the format used. Typical values
                include 'png' and 'jpg'.
            image_scale: Optional. Float. Default: 1.0. If the image_data
                is actually something that can be turned into an image,
                this will be the new scale of the image.
            image_shape: Optional. Tuple. Default: None. If the image_data
                is actually something that can be turned into an image,
                this is the new shape of the array. Dimensions are (width, height)
                or (width, height, colors) where colors is 3 (RGB) or 1 (grayscale).
            image_colormap: Optional. String. If the image_data is
                actually something that can be turned into an image,
                this is the colormap used to colorize the matrix.
            image_minmax: Optional. (Number, Number). If the image_data
                is actually something that can be turned into an image,
                this is the (min, max) used to scale the values.
                Otherwise, the image is autoscaled between (array.min, array.max).
            image_channels: Optional. Default 'last'. If the image_data is
                actually something that can be turned into an image,
                this is the setting that indicates where the color
                information is in the format of the 2D data. 'last'
                indicates that the data is in (rows, columns, channels)
                where 'first' indicates (channels, rows, columns).
            copy_to_tmp: If image_data is not a file path, then this flag
                determines if the image is first copied to a temporary
                file before upload. If copy_to_tmp is False, then it is
                sent directly to the cloud.
            step: Optional. Used to associate the image asset to a specific step.
        """
        self._experiment.log_image(image_data, name, overwrite,
                                   image_format, image_scale,
                                   image_shape, image_colormap,
                                   image_minmax, image_channels,
                                   copy_to_tmp, step)

    @_requiresComet
    def log_code(self, file_name=None, folder=None, code=None,
                 code_name=None):
        """Log additional source code files.
        Args:   
            file_name: optional, string: the file path of the file to log
            folder: optional, string: the folder path where the code files
                are stored
            code: optional, string: source code, either as a string or a
                file-like object (like StringIO). If passed, code_name is mandatory
            code_name: optional, string: name of the source code file when
                code parameter is passed
        """
        self._experiment.log_code(file_name, folder, code, code_name)

    @_requiresComet
    def log_asset(self, file_data, file_name=None, overwrite=False,
                  copy_to_tmp=True, step=None, metadata=None):
        """Logs the Asset determined by file_data.
        Args:  
        file_data: String or File-like - either the file path of the
            file you want to log, or a file-like asset.
        file_name: String - Optional. A custom file name to be displayed.
            If not provided the filename from the file_data argument will be used.
        overwrite: if True will overwrite all existing assets with
            the same name.
        copy_to_tmp: If file_data is a file-like object, then this flag
            determines if the file is first copied to a temporary file
            before upload. If copy_to_tmp is False, then it is sent
            directly to the cloud.
        step: Optional. Used to associate the asset to a specific step.
        """
        self._experiment.log_asset(file_data, file_name, overwrite,
                                   copy_to_tmp, step, metadata)

    @_requiresComet
    def add_tag(self, tag):
        """Add a tag to the experiment.
        Args:   
            tag: String. A tag to add to the experiment.
        """
        self._experiment.add_tag(tag)
        