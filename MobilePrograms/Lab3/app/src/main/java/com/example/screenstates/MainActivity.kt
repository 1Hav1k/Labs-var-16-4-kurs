package com.example.screenstates

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

// ---------------------------------------------------------------
// 1. КОРНЕВОЙ ЭКРАН: смотрит на состояние и решает, что показать
// ---------------------------------------------------------------
@Composable
fun ScreenRoot(viewModel: ScreenViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize()) {
        // --- ВРЕМЕННАЯ ПАНЕЛЬ УПРАВЛЕНИЯ ---
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Button(onClick = { viewModel.loadData() }, modifier = Modifier.weight(1f)) {
                Text("Контент", fontSize = 12.sp)
            }
            Button(onClick = { viewModel.loadData(forceEmpty = true) }, modifier = Modifier.weight(1f)) {
                Text("Пусто", fontSize = 12.sp)
            }
            Button(onClick = { viewModel.loadData(forceError = true) }, modifier = Modifier.weight(1f)) {
                Text("Ошибка", fontSize = 12.sp)
            }
        }

        // --- САМ ЭКРАН ---
        Box(modifier = Modifier.weight(1f)) {
            when (val s = state) {
                is ScreenState.Loading -> LoadingScreen()
                is ScreenState.Content -> ContentScreen(s.items)
                is ScreenState.Empty   -> EmptyScreen()
                is ScreenState.Error   -> ErrorScreen(s.message)
            }
        }
    }
}

// ---------------------------------------------------------------
// 2. LOADING
// ---------------------------------------------------------------
@Composable
fun LoadingScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator()
            Spacer(Modifier.height(16.dp))
            Text("Загрузка...", fontSize = 16.sp)
        }
    }
}

// ---------------------------------------------------------------
// 3. CONTENT
// ---------------------------------------------------------------
@Composable
fun ContentScreen(items: List<String>) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(items) { item ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = item,
                    modifier = Modifier.padding(16.dp),
                    fontSize = 18.sp
                )
            }
        }
    }
}

// ---------------------------------------------------------------
// 4. EMPTY
// ---------------------------------------------------------------
@Composable
fun EmptyScreen() {
    Box(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("📭", fontSize = 64.sp)
            Spacer(Modifier.height(16.dp))
            Text(
                text = "Здесь пока ничего нет",
                fontSize = 20.sp,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Добавьте первый элемент, чтобы начать",
                fontSize = 14.sp,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ---------------------------------------------------------------
// 5. ERROR
// ---------------------------------------------------------------
@Composable
fun ErrorScreen(message: String) {
    Box(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("⚠️", fontSize = 64.sp)
            Spacer(Modifier.height(16.dp))
            Text(
                text = "Что-то пошло не так",
                fontSize = 20.sp,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = message,
                fontSize = 14.sp,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.error
            )
        }
    }
}

// ---------------------------------------------------------------
// 6. ТОЧКА ВХОДА
// ---------------------------------------------------------------
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ScreenRoot()
                }
            }
        }
    }
}

// ---------------------------------------------------------------
// 7. ПРЕВЬЮ — по одному на каждое состояние
// ---------------------------------------------------------------
@Preview(showBackground = true, showSystemUi = true)
@Composable
fun LoadingPreview() {
    MaterialTheme { LoadingScreen() }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
fun ContentPreview() {
    MaterialTheme {
        ContentScreen(listOf("Первый элемент", "Второй элемент", "Третий элемент"))
    }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
fun EmptyPreview() {
    MaterialTheme { EmptyScreen() }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
fun ErrorPreview() {
    MaterialTheme { ErrorScreen("Не удалось загрузить данные") }
}