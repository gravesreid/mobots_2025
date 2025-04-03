from main import MoBot
import lgpio

mobot = MoBot(chip = lgpio.gpiochip_open(4))
mobot.stop()