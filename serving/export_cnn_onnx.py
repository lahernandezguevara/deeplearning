import torch 
import onnx
import onnxruntime as ort
from models.factory import create_model
# onnx -> Unifica la forma en que un modelo se guarda para mandarse a produccion, como estado de inferencia
# triton -> nvidia. Puedo entrenar con cuda y 

# ======================================================
# Guardar modelo
# ====================================================
def main() -> None:
    # Cambia num_classes por el número de clases X de tu dataset nuevo
    num_classes = 10
    model = create_model("vgg11", num_classes=num_classes)

    # Estado interno del modelo
    state_dict = torch.load(
        '/home/edgar/Dev/uni/university-7th-semester/deep_learning/project1/artifacts/best_model.tbh',
        map_location = 'cpu',
        weights_only=True,
    )

    # Este estado es el mejor modelo que pudimos haber entrenado
    model.load_state_dict(state_dict)

    # Desactivamos todo lo que usamos para entrenar
    model.eval()

    dummy_input = torch.randn(
        # Tensor con numeros aleatorios para probar que las dimensiones del modelo esten bien
        8, 3, 224, 224
    )

    # Exportando modelo a onnx (formato libre de nn)
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        input_names=['input'],
        output_names=['output'],
        dynamic_shapes={
            "x": {0: "batch_size"},
        },
        # torch export utilice un exportador moderno que utilice el rafo interno del modelo y lo exporta usando
        # un batch dinamico
        # Un servidor de inferencia tiene que ser capaz de implementar un batch con el unico limite que
        # sea la memoria de mi servidor
        dynamo=True
    )

    onnx_program.save(
        '/home/edgar/Dev/uni/university-7th-semester/deep_learning/project1/serving/model_repository/cnn/1/model.onnx'
    )

    onnx_model = onnx.load(
        '/home/edgar/Dev/uni/university-7th-semester/deep_learning/project1/serving/model_repository/cnn/1/model.onnx'
    )
# =================================================================================================
    ## Validar que el modelo guardado este bien
# =============================================================================
    onnx.checker.check_model(onnx_model)
    with torch.inference_mode():
        torch_output = model(dummy_input)

    # onnxruntime para poder ejecutar el modelo
    session = ort.InferenceSession(
        '/home/edgar/Dev/uni/university-7th-semester/deep_learning/project1/serving/model_repository/cnn/1/model.onnx',
        # Forzando a ejecutar con cpu
        providers=['CPUExecutionProvider']
    )

    onnx_output = session.run(
        ["output"], {'input': dummy_input.numpy()}
    )[0]

    # Toma la salida de torch, toma la salida de onnx, la transforma a tensor y 
    # damos una toleranci
    torch.testing.assert_close(
        torch_output,
        torch.tensor(onnx_output),
        rtol=1e-3,
        atol=1e-5,
    )

    print("Modelo ONNX exportado y validado !")

if __name__ == '__main__':
    main()


