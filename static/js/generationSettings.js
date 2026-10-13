// Temperature slider <-> stored value. The slider's first stop (one step below
// the real 0..2 range) means "Default": the model's own temperature, stored as null.
export const TEMPERATURE_MIN = 0;
export const TEMPERATURE_MAX = 2;
export const TEMPERATURE_STEP = 0.1;
export const TEMPERATURE_DEFAULT_STOP = TEMPERATURE_MIN - TEMPERATURE_STEP;

export function temperatureToSlider(stored) {
  return stored == null ? TEMPERATURE_DEFAULT_STOP : Number(stored);
}

export function sliderToTemperature(sliderValue) {
  const n = Number(sliderValue);
  return n < TEMPERATURE_MIN ? null : n;
}

export function formatTemperature(sliderValue) {
  return sliderToTemperature(sliderValue) === null ? 'Default' : Number(sliderValue).toFixed(1);
}
