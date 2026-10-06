package com.example.screenstates

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ScreenViewModel : ViewModel() {

    // Внутреннее изменяемое состояние
    private val _state = MutableStateFlow<ScreenState>(ScreenState.Loading)
    // Наружу отдаём только для чтения
    val state: StateFlow<ScreenState> = _state.asStateFlow()

    init {
        loadData()
    }

    /**
     * Имитация загрузки данных. Каждый третий вызов даёт ошибку,
     * чтобы можно было посмотреть все состояния.
     */
    fun loadData(forceEmpty: Boolean = false, forceError: Boolean = false) {
        viewModelScope.launch {
            _state.value = ScreenState.Loading
            delay(1500) // имитируем сетевой запрос

            _state.value = when {
                forceError -> ScreenState.Error("Не удалось загрузить данные. Проверьте соединение.")
                forceEmpty -> ScreenState.Empty
                else -> {
                    val items = listOf(
                        "Первый элемент",
                        "Второй элемент",
                        "Третий элемент",
                        "Четвёртый элемент"
                    )
                    ScreenState.Content(items)
                }
            }
        }
    }
}