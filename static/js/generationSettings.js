// Temperature slider <-> stored value. The slider's last stop (above the real
// 0..2 range) means "Default": the model's own temperature, stored as null.
export const TEMPERATURE_DEFAULT_STOP = 2.1;

export function temperatureToSlider(stored) {
  return stored == null ? TEMPERATURE_DEFAULT_STOP : Number(stored);
}

export function sliderToTemperature(sliderValue) {
  const n = Number(sliderValue);
  return n > 2 ? null : n;
}

export function formatTemperature(sliderValue) {
  return sliderToTemperature(sliderValue) === null ? 'Default' : Number(sliderValue).toFixed(1);
}
