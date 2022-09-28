#!!!!codetype circitPython

import board
import busio
import digitalio
from adafruit_esp32spi import adafruit_esp32spi
import time
import adafruit_dht
import adafruit_requests as requests
from adafruit_esp32spi import adafruit_esp32spi_socket

###
from secrets import secrets

##

INTERVAL = 10
state = 0
threshold = 35

s = secrets()
ssid = s.ssid
password = s.pw


# FeatherWing ESP32 AirLift, nRF52840
cs = digitalio.DigitalInOut(board.D13)
rdy = digitalio.DigitalInOut(board.D11)
rst = digitalio.DigitalInOut(board.D12)

spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
wifi = adafruit_esp32spi.ESP_SPIcontrol(spi, cs, rdy, rst)
requests.set_socket(adafruit_esp32spi_socket, wifi)

while not wifi.is_connected:
    print("\nConnecting...")
    try:
        wifi.connect_AP(ssid, password)
    except RuntimeError as e:
        print("Cannot connect", e)
        continue

print("Connected to", str(wifi.ssid, "utf-8"))
print("IP address", wifi.pretty_ip(wifi.ip_address))

# Setup #
# Sensoren
dht = adafruit_dht.DHT11(board.D9)  # nRF52840, Grove D4
# LED
led = digitalio.DigitalInOut(board.A4)  # general-purpose RED LED on Pin D3
led.direction = digitalio.Direction.OUTPUT
# Button
button = digitalio.DigitalInOut(board.D5)  # nRF52840, Grove D2
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.DOWN
# relais A0
relais = digitalio.DigitalInOut(board.A0)
relais.direction = digitalio.Direction.OUTPUT

while True:
    start = time.time()
    t = time.localtime(start)
    try:
        # Read the temperature and convert it to integer
        temperature = int(round(dht.temperature))
        # Read the humidity and convert it to integer
        humidity = int(round(dht.humidity))

        # check conditions, set new values, next state
        if (state == 0) and button.value:
            relais.value = True
            led.value = True
            t0 = time.monotonic()
            print(t0)
            state = 2
            print(f"{state}")
        elif (state == 0) and (humidity < threshold):
            relais.value = True
            led.value = True
            state = 1
            print(f"{state}")

        elif (state == 1) and (humidity >= threshold):
            relais.value = False
            led.value = False
            state = 0
            print(f"{state}")

        elif (state == 2) and not button.value:
            state = 3
            print(f"{state}")

        elif (state == 3) and ((t0 + 3600) <= time.monotonic()):
            relais.value = False
            led.value = False
            state = 0
            print(f"{state}")

        # read sensors
        # check conditions, set new values, next state
        # Print timestamp, temperatur, humidity
        result = "{:d}:{:02d}:{:02d},{:g},{:g}".format(
            t.tm_hour, t.tm_min, t.tm_sec, temperature, humidity)
        print(result)
        # create url
        date_string = "{:d}:{:02d}:{:02d}".format(
                       t.tm_hour, t.tm_min, t.tm_sec)

        post_url = "https://localhost:5000/fill:" + str(temperature) + str(humidity) + date_string



        response = requests.get(post_url)
        # print(response.status_code)

    except RuntimeError as e:
        # Reading doesn't always work! Just print error and we'll try again
        print("{:d}:{:02d}:{:02d},{:g},{:g}".format(
            t.tm_hour, t.tm_min, t.tm_sec, -1, -1))

    end = time.time()
    # Wait for the remaining time
    time.sleep(INTERVAL - (end - start))
