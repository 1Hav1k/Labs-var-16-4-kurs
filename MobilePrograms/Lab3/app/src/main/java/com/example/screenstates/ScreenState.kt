package com.example.screenstates

sealed interface ScreenState {
    object Loading : ScreenState
    data class Content(val items: List<String>) : ScreenState
    object Empty : ScreenState
    data class Error(val message: String) : ScreenState
}