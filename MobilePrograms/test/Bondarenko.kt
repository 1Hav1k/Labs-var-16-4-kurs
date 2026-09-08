import kotlin.system.exitProcess

class Bondarenko {
    private var count = 0

    fun drinkE() {
        count++
        println("Энергетик выпит")
        if (count > 3) {
            println("Предупреждение: выпито уже $count энергетиков!")
        }
        if (count > 5) {
            println("Выпито слишком много энергетиков")
            exitProcess(0)
        }
    }
}

fun main() {
    val bondarenko = Bondarenko()
    while (true) {
        println("Выберите действие:")
        println("1. Выпить энергетик")
        println("2. Выйти из программы")
        when (readLine()?.trim()) {
            "1" -> bondarenko.drinkE()
            "2" -> {
                println("Выход из программы.")
                return
            }
            else -> println("Неверный ввод, попробуйте снова.")
        }
    }
}