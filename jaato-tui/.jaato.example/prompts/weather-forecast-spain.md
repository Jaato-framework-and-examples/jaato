---
description: Get a detailed weather forecast for a Spanish city
params:
  city:
    required: true
    description: Spanish city name (e.g., Madrid, Barcelona, Sevilla, Valencia, Bilbao)
  days:
    required: false
    default: "3"
    description: Number of forecast days (1-7)
  language:
    required: false
    default: Spanish
    description: Response language (Spanish or English)
tags: ['weather', 'spain', 'utility']
---

Search the web for the current weather forecast for **{{city}}, Spain** for the
next **{{days}} days**.

Present the forecast in **{{language}}** using this structure:

1. **Current conditions**: Temperature, humidity, wind speed, sky conditions.
2. **Day-by-day forecast**: For each of the next {{days}} days, include:
   - Date and day of the week
   - High and low temperatures (Celsius)
   - Precipitation probability
   - General conditions (sunny, cloudy, rain, etc.)
   - Wind speed and direction
3. **Clothing recommendation**: Based on the forecast, suggest what to wear
   (jacket, umbrella, sunscreen, etc.)
4. **Local tip**: Mention one outdoor activity or place to visit in {{city}}
   that suits the forecasted weather.

Use the metric system (Celsius, km/h). Be concise but informative.
