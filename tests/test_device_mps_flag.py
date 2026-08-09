from evalaware.models.device import enable_mps_fallback, get_device

def test_device_info_has_fallback_field():
    info = get_device(None)
    d = info.to_dict()
    assert "mps_fallback_enabled" in d

def test_enable_mps_fallback_bool():
    assert isinstance(enable_mps_fallback(), bool)
