window.HELP_IMPROVE_VIDEOJS = false;

function createInterpolationSection(wrapperId, sliderId, basePath, numFrames) {
  var interpImages = [];

  function preloadInterpolationImages() {
    for (var i = 0; i < numFrames; i++) {
      var path = basePath + '/' + String(i).padStart(4, '0') + '.png';
      interpImages[i] = new Image();
      interpImages[i].src = path;
    }
  }

  function setInterpolationImage(i) {
    var image = interpImages[i];
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#' + wrapperId).empty().append(image);
  }

  // Preload images and set up slider behavior
  $(document).ready(function() {
    preloadInterpolationImages();

    $('#' + sliderId).on('input', function(event) {
      setInterpolationImage(this.value);
    });

    setInterpolationImage(0);
    $('#' + sliderId).prop('max', numFrames - 1);
  });
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    createInterpolationSection('interpolation-image-wrapper-1', 'interpolation-slider-1', './static/interpolation/demo_video_1', 108);

    createInterpolationSection('interpolation-image-wrapper-2', 'interpolation-slider-2', './static/interpolation/demo_video_2', 107);

    createInterpolationSection('interpolation-image-wrapper-3', 'interpolation-slider-3', './static/interpolation/demo_video_3', 92);

    createInterpolationSection('interpolation-image-wrapper-4', 'interpolation-slider-4', './static/interpolation/demo_video_4', 119);

    bulmaSlider.attach();

})
