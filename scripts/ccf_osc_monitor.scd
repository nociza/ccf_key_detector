// CCF Tonal PDF OSC monitor
// Listens for /ccf/pdf messages (with optional hop index) and renders them as a bar graph.

(
var window, sliders, infoLabel, port = 7400;

// Clean up any previous instance.
OSCdef(\ccfTonalMonitor).free;

window = Window("CCF Tonal PDF", Rect(100, 100, 720, 320))
    .front
    .alwaysOnTop_(true);

sliders = MultiSliderView(window, Rect(10, 10, 700, 220));
sliders.elasticMode_(1);
sliders.isFilled_(true);
sliders.thumbSize_(0.0);
sliders.value_([0.0]);

infoLabel = StaticText(window, Rect(10, 240, 700, 20))
    .align_(\left)
    .string_("Waiting for /ccf/pdf …");

OSCdef(\ccfTonalMonitor,
    { |msg, time, addr, recvPort|
        {
            var payload = msg.drop(1);
            var frameIndex;
            var pdf;

            if(payload.notEmpty and: { payload[0].isKindOf(Integer) and: { payload.size > 1 } }) {
                frameIndex = payload[0].asInteger;
                pdf = payload.drop(1);
            } {
                frameIndex = nil;
                pdf = payload;
            };

            pdf = pdf.collect(_.asFloat);

            // Resize if n_bins changes at runtime.
            if(pdf.size != sliders.value.size) {
                sliders.value_(Array.fill(pdf.size, 0.0));
            };

            sliders.value_(pdf);

            if(pdf.notEmpty) {
                var peak = pdf.maxIndex;
                var peakVal = pdf[peak];
                infoLabel.string_(
                    frameIndex.notNil.if(
                        { ("frame % | peak bin % | likelihood %.3f").format(frameIndex, peak, peakVal) },
                        { ("peak bin % | likelihood %.3f").format(peak, peakVal) }
                    )
                );
            };
        }.defer;  // Ensure GUI updates run on the AppClock
    },
    '/ccf/pdf',
    recvPort: port
);

window.onClose = {
    OSCdef(\ccfTonalMonitor).free;
};
)
